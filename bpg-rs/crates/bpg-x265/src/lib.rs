//! Safe wrapper around the vendored x265, implementing
//! [`bpg_encode::HevcEncoder`]. Ported from `x265_glue.c` in
//! `libbpg-0.9.8` (the `open`/`encode`/`close` sequence), specialized to the
//! M1 still-image, single-frame case.

use std::os::raw::{c_int, c_void};
use std::ptr;

use bpg_encode::{EncodeError, HevcEncodeParams, HevcEncoder};
use bpg_image::{ChromaFormat, Image};
use bpg_x265_sys as sys;

/// x265 rate-control mode enum (anonymous in `x265.h`, so not emitted by
/// bindgen): `ABR = 0, CQP = 1, CRF = 2`.
const X265_RC_CQP: c_int = 1;

/// x265 preset names indexed by compress level (1 = fast .. 9 = placebo).
/// `static const` in `x265.h`, so not emitted by bindgen. Index 0 is
/// "ultrafast"; the BPG default compress level (8) selects "veryslow".
const X265_PRESET_NAMES: [&str; 10] = [
    "ultrafast\0",
    "superfast\0",
    "veryfast\0",
    "faster\0",
    "fast\0",
    "medium\0",
    "slow\0",
    "slower\0",
    "veryslow\0",
    "placebo\0",
];

const TUNE_SSIM: &str = "ssim\0";

/// The x265 HEVC backend.
pub struct X265Encoder;

impl X265Encoder {
    pub fn new() -> Self {
        X265Encoder
    }
}

impl Default for X265Encoder {
    fn default() -> Self {
        Self::new()
    }
}

fn csp_of(cf: ChromaFormat) -> c_int {
    match cf {
        ChromaFormat::Gray => sys::X265_CSP_I400 as c_int,
        ChromaFormat::Yuv420 => sys::X265_CSP_I420 as c_int,
        ChromaFormat::Yuv422 => sys::X265_CSP_I422 as c_int,
        ChromaFormat::Yuv444 => sys::X265_CSP_I444 as c_int,
    }
}

impl HevcEncoder for X265Encoder {
    fn encode_still(
        &self,
        image: &Image,
        params: &HevcEncodeParams,
    ) -> Result<Vec<u8>, EncodeError> {
        // SAFETY: every raw pointer below comes from the x265 API and is
        // checked for null; `image`'s plane buffers outlive the encode call.
        unsafe { encode_impl(image, params) }
    }
}

unsafe fn encode_impl(
    image: &Image,
    params: &HevcEncodeParams,
) -> Result<Vec<u8>, EncodeError> {
    let mut err: c_int = 0;
    let api = sys::x265_api_query(params.bit_depth as c_int, sys::X265_BUILD as c_int, &mut err);
    if api.is_null() {
        return Err(EncodeError::Backend(format!(
            "x265_api_query failed (bit_depth={}, err={})",
            params.bit_depth, err
        )));
    }
    let api = &*api;

    let macro_err = |s: &str| EncodeError::Backend(s.to_string());

    let param_alloc = api.param_alloc.ok_or_else(|| macro_err("param_alloc null"))?;
    let param_free = api.param_free.ok_or_else(|| macro_err("param_free null"))?;
    let param_default_preset = api
        .param_default_preset
        .ok_or_else(|| macro_err("param_default_preset null"))?;
    let picture_alloc = api.picture_alloc.ok_or_else(|| macro_err("picture_alloc null"))?;
    let picture_init = api.picture_init.ok_or_else(|| macro_err("picture_init null"))?;
    let picture_free = api.picture_free.ok_or_else(|| macro_err("picture_free null"))?;
    let encoder_open = api.encoder_open.ok_or_else(|| macro_err("encoder_open null"))?;
    let encoder_encode = api.encoder_encode.ok_or_else(|| macro_err("encoder_encode null"))?;
    let encoder_close = api.encoder_close.ok_or_else(|| macro_err("encoder_close null"))?;

    let p = param_alloc();
    if p.is_null() {
        return Err(macro_err("param_alloc returned null"));
    }

    let preset_index = (params.compress_level as usize).min(X265_PRESET_NAMES.len() - 1);
    let preset = X265_PRESET_NAMES[preset_index];
    param_default_preset(
        p,
        preset.as_ptr() as *const _,
        TUNE_SSIM.as_ptr() as *const _,
    );

    // Mirror x265_open() in x265_glue.c.
    (*p).bRepeatHeaders = 1;
    (*p).decodedPictureHashSEI = params.sei_decoded_picture_hash as c_int;
    (*p).sourceWidth = params.width as c_int;
    (*p).sourceHeight = params.height as c_int;
    (*p).internalCsp = csp_of(params.chroma_format);
    // intra_only (still image): I frames only.
    (*p).keyframeMax = 1;
    (*p).totalFrames = 1;
    (*p).bEnableRectInter = 1;
    (*p).bEnableAMP = 1; /* required: header restriction forbids amp_enabled_flag==0 */
    (*p).internalBitDepth = params.bit_depth as c_int;
    (*p).bEmitInfoSEI = 0;
    (*p).logLevel = if params.verbose {
        sys::X265_LOG_INFO as c_int
    } else {
        sys::X265_LOG_NONE
    };
    /* dummy frame rate */
    (*p).fpsNum = 25;
    (*p).fpsDenom = 1;
    (*p).rc.rateControlMode = X265_RC_CQP;
    (*p).rc.qp = params.qp as c_int;
    (*p).bLossless = 0; /* TODO(extension): lossless mode */

    let enc = encoder_open(p);
    if enc.is_null() {
        param_free(p);
        return Err(macro_err("encoder_open failed"));
    }

    let pic = picture_alloc();
    if pic.is_null() {
        encoder_close(enc);
        param_free(p);
        return Err(macro_err("picture_alloc failed"));
    }
    picture_init(p, pic);
    (*pic).colorSpace = (*p).internalCsp;

    let c_count = if params.chroma_format == ChromaFormat::Gray {
        1
    } else {
        3
    };
    if image.planes.len() < c_count {
        picture_free(pic);
        encoder_close(enc);
        param_free(p);
        return Err(macro_err("image has too few planes for chroma format"));
    }
    // `Image` always stores samples as `u16` (matching `bpgenc.c`'s internal
    // `PIXEL = uint16_t`). x265's 8-bit build expects 1-byte `uint8_t`
    // samples (mirroring `image_convert16to8`); 10/12-bit builds expect
    // 2-byte `uint16_t` samples directly, with `stride` in bytes.
    let mut u8_planes: Vec<Vec<u8>> = Vec::new();
    if image.bit_depth == 8 {
        for i in 0..c_count {
            u8_planes.push(image.planes[i].data.iter().map(|&v| v as u8).collect());
        }
        for i in 0..c_count {
            (*pic).planes[i] = u8_planes[i].as_ptr() as *mut c_void;
            (*pic).stride[i] = image.planes[i].stride as c_int;
        }
    } else {
        for i in 0..c_count {
            (*pic).planes[i] = image.planes[i].data.as_ptr() as *mut c_void;
            (*pic).stride[i] = (image.planes[i].stride * 2) as c_int;
        }
    }
    (*pic).bitDepth = image.bit_depth as c_int;

    let mut out: Vec<u8> = Vec::new();
    let mut collect = |p_nal: *mut sys::x265_nal, nal_count: u32| {
        for i in 0..nal_count as isize {
            let nal = &*p_nal.offset(i);
            let slice = std::slice::from_raw_parts(nal.payload, nal.sizeBytes as usize);
            out.extend_from_slice(slice);
        }
    };

    // Encode the single frame.
    let mut p_nal: *mut sys::x265_nal = ptr::null_mut();
    let mut nal_count: u32 = 0;
    let ret = encoder_encode(enc, &mut p_nal, &mut nal_count, pic, ptr::null_mut());
    if ret > 0 {
        collect(p_nal, nal_count);
    }

    // Flush any buffered output.
    loop {
        let ret = encoder_encode(enc, &mut p_nal, &mut nal_count, ptr::null_mut(), ptr::null_mut());
        if ret <= 0 {
            break;
        }
        collect(p_nal, nal_count);
    }

    picture_free(pic);
    encoder_close(enc);
    param_free(p);

    if out.is_empty() {
        return Err(macro_err("x265 produced no output NALs"));
    }
    Ok(out)
}
