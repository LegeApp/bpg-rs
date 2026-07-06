//! Decoded frame representation

use alloc::vec;
use alloc::vec::Vec;

use super::color_convert;

/// Sentinel value for uninitialized pixels.
/// Used during decoding to distinguish decoded samples from uninitialized ones
/// for reference sample availability (H.265 8.4.4.2.2).
pub const UNINIT_SAMPLE: u16 = u16::MAX;

/// Deblocking edge flags per 4x4 block
pub const DEBLOCK_FLAG_VERT: u8 = 1;
/// Horizontal edge flag
pub const DEBLOCK_FLAG_HORIZ: u8 = 2;

/// Decoded video frame
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Width in pixels (full frame, before cropping)
    pub width: u32,
    /// Height in pixels (full frame, before cropping)
    pub height: u32,
    /// Luma (Y) plane
    pub y_plane: Vec<u16>,
    /// Cb chroma plane (half resolution for 4:2:0)
    pub cb_plane: Vec<u16>,
    /// Cr chroma plane (half resolution for 4:2:0)
    pub cr_plane: Vec<u16>,
    /// Bit depth
    pub bit_depth: u8,
    /// Chroma format (1=4:2:0, 2=4:2:2, 3=4:4:4)
    pub chroma_format: u8,
    /// Conformance window left offset (in luma samples)
    pub crop_left: u32,
    /// Conformance window right offset (in luma samples)
    pub crop_right: u32,
    /// Conformance window top offset (in luma samples)
    pub crop_top: u32,
    /// Conformance window bottom offset (in luma samples)
    pub crop_bottom: u32,
    /// Deblocking edge flags at 4x4 block granularity
    /// Bit 0 = vertical edge, Bit 1 = horizontal edge
    pub deblock_flags: Vec<u8>,
    /// Stride for deblock_flags (width / 4)
    pub deblock_stride: u32,
    /// QP map at 4x4 block granularity (for deblocking)
    pub qp_map: Vec<i8>,
    /// Alpha plane (optional, from auxiliary alpha image)
    pub alpha_plane: Option<Vec<u16>>,
    /// Video full range flag (from SPS VUI). true = full \[0,255\], false = limited \[16,235\]
    pub full_range: bool,
    /// Matrix coefficients (from SPS VUI). 1=BT.709, 5/6=BT.601, 9=BT.2020, 2=unspecified
    pub matrix_coeffs: u8,
}

impl DecodedFrame {
    /// Create a new frame buffer
    ///
    /// # Panics
    /// Panics if width * height overflows u32.
    pub fn new(width: u32, height: u32) -> Self {
        let luma_size = width
            .checked_mul(height)
            .expect("frame dimensions overflow") as usize;
        // Assume 4:2:0 chroma subsampling
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let chroma_size = (chroma_width * chroma_height) as usize;
        let deblock_stride = width.div_ceil(4);
        let deblock_height = height.div_ceil(4);
        let deblock_size = (deblock_stride * deblock_height) as usize;

        Self {
            width,
            height,
            y_plane: vec![UNINIT_SAMPLE; luma_size],
            cb_plane: vec![UNINIT_SAMPLE; chroma_size],
            cr_plane: vec![UNINIT_SAMPLE; chroma_size],
            bit_depth: 8,
            chroma_format: 1, // 4:2:0
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            deblock_flags: vec![0; deblock_size],
            deblock_stride,
            qp_map: vec![0; deblock_size],
            alpha_plane: None,
            full_range: false,
            matrix_coeffs: 2,
        }
    }

    /// Create a frame with specific parameters
    ///
    /// # Panics
    /// Panics if width * height overflows u32.
    pub fn with_params(width: u32, height: u32, bit_depth: u8, chroma_format: u8) -> Self {
        Self::with_params_filled(width, height, bit_depth, chroma_format, UNINIT_SAMPLE)
    }

    /// Like [`Self::with_params`], but reconstruction samples start at
    /// `fill`. Passing `0` yields lazily-zeroed allocations, so untouched
    /// regions cost neither fill time nor physical pages — useful for
    /// buffers whose every read position is written first (e.g. encoder WPP
    /// worker frames).
    ///
    /// # Panics
    /// Panics if width * height overflows u32.
    pub fn with_params_filled(
        width: u32,
        height: u32,
        bit_depth: u8,
        chroma_format: u8,
        fill: u16,
    ) -> Self {
        let luma_size = width
            .checked_mul(height)
            .expect("frame dimensions overflow") as usize;

        let (chroma_width, chroma_height) = match chroma_format {
            0 => (0, 0),                                  // Monochrome
            1 => (width.div_ceil(2), height.div_ceil(2)), // 4:2:0
            2 => (width.div_ceil(2), height),             // 4:2:2
            3 => (width, height),                         // 4:4:4
            _ => (width.div_ceil(2), height.div_ceil(2)),
        };

        let chroma_size = (chroma_width * chroma_height) as usize;

        let deblock_stride = width.div_ceil(4);
        let deblock_height = height.div_ceil(4);
        let deblock_size = (deblock_stride * deblock_height) as usize;

        Self {
            width,
            height,
            y_plane: vec![fill; luma_size],
            cb_plane: vec![fill; chroma_size],
            cr_plane: vec![fill; chroma_size],
            bit_depth,
            chroma_format,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            deblock_flags: vec![0; deblock_size],
            deblock_stride,
            qp_map: vec![0; deblock_size],
            alpha_plane: None,
            full_range: false,
            matrix_coeffs: 2,
        }
    }

    /// Mark a vertical TU/CU boundary at luma position (x, y) with given size
    pub fn mark_tu_boundary(&mut self, x: u32, y: u32, size: u32) {
        let bx = x / 4;
        let by = y / 4;
        let bs = size / 4;

        // Mark vertical edge at x (left edge of TU)
        if x > 0 {
            for j in 0..bs {
                let idx = ((by + j) * self.deblock_stride + bx) as usize;
                if idx < self.deblock_flags.len() {
                    self.deblock_flags[idx] |= DEBLOCK_FLAG_VERT;
                }
            }
        }

        // Mark horizontal edge at y (top edge of TU)
        if y > 0 {
            for i in 0..bs {
                let idx = (by * self.deblock_stride + bx + i) as usize;
                if idx < self.deblock_flags.len() {
                    self.deblock_flags[idx] |= DEBLOCK_FLAG_HORIZ;
                }
            }
        }
    }

    /// Store QP for a block region at 4x4 granularity
    pub fn store_block_qp(&mut self, x: u32, y: u32, size: u32, qp: i8) {
        let bx = x / 4;
        let by = y / 4;
        let bs = size / 4;
        for j in 0..bs {
            for i in 0..bs {
                let idx = ((by + j) * self.deblock_stride + bx + i) as usize;
                if idx < self.qp_map.len() {
                    self.qp_map[idx] = qp;
                }
            }
        }
    }

    /// Set conformance window cropping
    pub fn set_crop(&mut self, left: u32, right: u32, top: u32, bottom: u32) {
        self.crop_left = left;
        self.crop_right = right;
        self.crop_top = top;
        self.crop_bottom = bottom;
    }

    /// Get cropped width
    pub fn cropped_width(&self) -> u32 {
        self.width - self.crop_left - self.crop_right
    }

    /// Get cropped height
    pub fn cropped_height(&self) -> u32 {
        self.height - self.crop_top - self.crop_bottom
    }

    /// Get luma stride (width)
    pub fn y_stride(&self) -> usize {
        self.width as usize
    }

    /// Get chroma stride
    pub fn c_stride(&self) -> usize {
        match self.chroma_format {
            0 => 0,
            1 | 2 => self.width.div_ceil(2) as usize,
            3 => self.width as usize,
            _ => self.width.div_ceil(2) as usize,
        }
    }

    /// Chroma subsampling factors `(SubWidthC, SubHeightC)` for this frame's
    /// chroma format (4:2:0 -> (2,2), 4:2:2 -> (2,1), 4:4:4/mono -> (1,1)).
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1), // 4:4:4 and monochrome/fallback
        }
    }

    /// Dimensions of the component plane `c_idx` (0 = luma) in samples. Chroma
    /// planes use ceil division so odd luma sizes round up, matching
    /// `with_params`' allocation.
    pub fn component_dims(&self, c_idx: u8) -> (u32, u32) {
        if c_idx == 0 || self.chroma_format == 0 {
            (self.width, self.height)
        } else {
            let (sw, sh) = self.chroma_subsampling();
            (self.width.div_ceil(sw), self.height.div_ceil(sh))
        }
    }

    /// Convert a single YCbCr pixel to RGB.
    /// y_val, cb_val, cr_val are 8-bit values (0-255).
    /// Selects coefficient matrix based on `matrix_coeffs` field.
    ///
    /// Both full-range and limited-range use integer fixed-point arithmetic.
    /// Full-range: ×256, limited-range: ×2048 with combined Y/C scale factors.
    #[inline(always)]
    fn ycbcr_to_rgb(&self, y_val: i32, cb_val: i32, cr_val: i32) -> (u8, u8, u8) {
        color_convert::pixel_to_rgb(
            y_val,
            cb_val,
            cr_val,
            self.bit_depth,
            self.full_range,
            self.matrix_coeffs,
        )
    }

    /// Lanczos-upsample subsampled chroma (4:2:0/4:2:2) to full luma resolution,
    /// returning 8-bit `(cb, cr)` planes of `width * height` samples. Returns
    /// `None` for 4:4:4 and monochrome (no upsampling needed). `shift` is
    /// `bit_depth - 8` so the result is always 8-bit. Shared by RGB output
    /// (`write_pixels`) and YCbCr output (`to_ycbcr444_8bit`) so both use the
    /// same bit-exact kernels.
    fn upsampled_full_chroma(&self, shift: u8) -> Option<(Vec<u16>, Vec<u16>)> {
        let w = self.width as usize;
        let h = self.height as usize;
        match self.chroma_format {
            1 => {
                let c_stride = self.c_stride();
                let w2 = (self.width as usize + 1) / 2;
                let h2 = (self.height as usize + 1) / 2;
                Some((
                    color_convert::upsample_chroma_420(
                        &self.cb_plane,
                        w2,
                        h2,
                        c_stride,
                        w,
                        h,
                        shift as u32,
                    ),
                    color_convert::upsample_chroma_420(
                        &self.cr_plane,
                        w2,
                        h2,
                        c_stride,
                        w,
                        h,
                        shift as u32,
                    ),
                ))
            }
            2 => {
                let c_stride = self.c_stride();
                let w2 = (self.width as usize + 1) / 2;
                Some((
                    color_convert::upsample_chroma_422(
                        &self.cb_plane,
                        w2,
                        h,
                        c_stride,
                        w,
                        h,
                        shift as u32,
                    ),
                    color_convert::upsample_chroma_422(
                        &self.cr_plane,
                        w2,
                        h,
                        c_stride,
                        w,
                        h,
                        shift as u32,
                    ),
                ))
            }
            _ => None,
        }
    }

    /// Render the cropped picture as interleaved 8-bit `[Y, Cb, Cr]` at full
    /// 4:4:4 resolution (chroma Lanczos-upsampled, bit-exact with the RGB path),
    /// **without** any YCbCr→RGB conversion. This is the color-space-preserving
    /// path for re-encoding to JPEG: when the frame is BT.601 full-range 8-bit,
    /// these samples can be written straight into a JFIF JPEG. The stored planes
    /// are not mutated. Monochrome frames get neutral chroma (128).
    pub fn to_ycbcr444_8bit(&self) -> Vec<u8> {
        let shift = self.bit_depth - 8;
        let w = self.width as usize;
        let cw = self.cropped_width() as usize;
        let ch = self.cropped_height() as usize;
        let full = self.upsampled_full_chroma(shift);
        let mut out = vec![0u8; cw * ch * 3];
        let y_start = self.crop_top;
        let x_start = self.crop_left;
        let x_end = self.width - self.crop_right;
        color_convert::for_each_row(&mut out, cw * 3, |row, out_row| {
            let y = y_start + row as u32;
            let mut off = 0usize;
            for x in x_start..x_end {
                let idx = y as usize * w + x as usize;
                let y_val = (self.y_plane[idx] >> shift).min(255) as u8;
                let (cb, cr) = match &full {
                    Some((cb, cr)) => (cb[idx].min(255) as u8, cr[idx].min(255) as u8),
                    None => {
                        let (cb, cr) = self.get_chroma(x, y, shift);
                        (cb.clamp(0, 255) as u8, cr.clamp(0, 255) as u8)
                    }
                };
                out_row[off] = y_val;
                out_row[off + 1] = cb;
                out_row[off + 2] = cr;
                off += 3;
            }
        });
        out
    }

    /// Render the cropped luma plane as 8-bit grayscale (`cropped_width *
    /// cropped_height` bytes), for writing monochrome frames to JPEG without
    /// going through RGB.
    pub fn to_luma8(&self) -> Vec<u8> {
        let shift = self.bit_depth - 8;
        let w = self.width as usize;
        let cw = self.cropped_width() as usize;
        let ch = self.cropped_height() as usize;
        let mut out = vec![0u8; cw * ch];
        let mut off = 0usize;
        for y in self.crop_top..(self.height - self.crop_bottom) {
            for x in self.crop_left..(self.width - self.crop_right) {
                let idx = y as usize * w + x as usize;
                out[off] = (self.y_plane[idx] >> shift).min(255) as u8;
                off += 1;
            }
        }
        out
    }

    /// Like [`to_luma8`] but expands a limited-range luma plane to full range
    /// (JFIF expects full-range luma). For full-range 8-bit input it equals
    /// [`to_luma8`]. Used for the grayscale JPEG path so non-full-range mono
    /// frames keep correct tones without an RGB conversion.
    pub fn to_luma8_jfif(&self) -> Vec<u8> {
        let w = self.width as usize;
        let cw = self.cropped_width() as usize;
        let ch = self.cropped_height() as usize;
        let bd = self.bit_depth;
        let fr = self.full_range;
        let mut out = vec![0u8; cw * ch];
        let mut off = 0usize;
        for y in self.crop_top..(self.height - self.crop_bottom) {
            for x in self.crop_left..(self.width - self.crop_right) {
                let idx = y as usize * w + x as usize;
                out[off] = color_convert::luma_to_jfif_8bit(self.y_plane[idx] as i32, bd, fr);
                off += 1;
            }
        }
        out
    }

    /// Render the cropped picture as interleaved 8-bit `[Y, Cb, Cr]` transcoded
    /// to **JFIF (BT.601 full-range)** YCbCr — the baseline JPEG color space —
    /// for sources that are not already BT.601 full-range (BT.709/BT.2020 or
    /// limited range). Chroma is Lanczos-upsampled to 4:4:4 first, then each
    /// pixel is converted in the YCbCr domain via
    /// [`color_convert::transcode_to_jfif_ycbcr`]; no clamped RGB image is ever
    /// formed. Only valid when `is_ycbcr_matrix(self.matrix_coeffs)`.
    pub fn to_jfif_ycbcr444_8bit(&self) -> Vec<u8> {
        let shift = self.bit_depth - 8;
        let w = self.width as usize;
        let cw = self.cropped_width() as usize;
        let ch = self.cropped_height() as usize;
        let full = self.upsampled_full_chroma(shift);
        let mc = self.matrix_coeffs;
        let fr = self.full_range;
        let y_start = self.crop_top;
        let x_start = self.crop_left;
        let x_end = self.width - self.crop_right;
        let mut out = vec![0u8; cw * ch * 3];
        color_convert::for_each_row(&mut out, cw * 3, |row, out_row| {
            let y = y_start + row as u32;
            let mut off = 0usize;
            for x in x_start..x_end {
                let idx = y as usize * w + x as usize;
                let y8 = (self.y_plane[idx] >> shift).min(255) as i32;
                let (cb8, cr8) = match &full {
                    Some((cb, cr)) => (cb[idx].min(255) as i32, cr[idx].min(255) as i32),
                    None => self.get_chroma(x, y, shift),
                };
                let (yy, cb, cr) = color_convert::transcode_to_jfif_ycbcr(y8, cb8, cr8, fr, mc);
                out_row[off] = yy;
                out_row[off + 1] = cb;
                out_row[off + 2] = cr;
                off += 3;
            }
        });
        out
    }

    /// Render the cropped picture into `out` as interleaved 8-bit pixels of
    /// `stride` bytes each; `r_pos`/`g_pos`/`b_pos` (and optional `a_pos`) give the
    /// byte offset of each channel within a pixel. For subsampled 4:2:0/4:2:2
    /// formats, chroma is Lanczos-upsampled to luma resolution (bit-exact with
    /// bpgdec) on the fly; the stored YCbCr planes are never mutated, so native
    /// chroma stays available for non-RGB consumers.
    fn write_pixels(
        &self,
        out: &mut [u8],
        stride: usize,
        r_pos: usize,
        g_pos: usize,
        b_pos: usize,
        a_pos: Option<usize>,
    ) {
        let shift = self.bit_depth - 8;
        let y_start = self.crop_top;
        let x_start = self.crop_left;
        let x_end = self.width - self.crop_right;
        let w = self.width as usize;

        // Subsampled chroma: upsample to full resolution once, separate from the
        // native planes (preserving unmodified YCbCr for other consumers).
        let full = self.upsampled_full_chroma(shift);
        let cw = (x_end - x_start) as usize;

        // YCbCr→RGB is a cross-channel per-pixel conversion; the win comes from
        // spreading whole output rows across cores (see `for_each_row`), which is
        // bit-identical to the sequential pass.
        color_convert::for_each_row(out, cw * stride, |row, out_row| {
            let y = y_start + row as u32;
            let mut off = 0usize;
            for (col, x) in (x_start..x_end).enumerate() {
                let idx = y as usize * w + x as usize;
                let y_val = (self.y_plane[idx] >> shift) as i32;
                let (cb_val, cr_val) = match &full {
                    Some((cb, cr)) => (cb[idx] as i32, cr[idx] as i32),
                    None => self.get_chroma(x, y, shift),
                };
                let (r, g, b) = self.ycbcr_to_rgb(y_val, cb_val, cr_val);
                out_row[off + r_pos] = r;
                out_row[off + g_pos] = g;
                out_row[off + b_pos] = b;
                if let Some(ap) = a_pos {
                    let pixel_idx = row * cw + col;
                    let alpha = match self.alpha_plane {
                        Some(ref a) if pixel_idx < a.len() => {
                            (a[pixel_idx] >> shift).min(255) as u8
                        }
                        _ => 255,
                    };
                    out_row[off + ap] = alpha;
                }
                off += stride;
            }
        });
    }

    /// Convert YCbCr to RGB with conformance window cropping.
    pub fn to_rgb(&self) -> Vec<u8> {
        let total = (self.cropped_width() * self.cropped_height()) as usize;
        let mut rgb = vec![0u8; total * 3];
        self.write_pixels(&mut rgb, 3, 0, 1, 2, None);
        rgb
    }

    /// Convert YCbCr to BGRA with conformance window cropping (blue, green, red,
    /// alpha). Uses real alpha from `alpha_plane` if present, otherwise 255.
    pub fn to_bgra(&self) -> Vec<u8> {
        let total = (self.cropped_width() * self.cropped_height()) as usize;
        let mut bgra = vec![0u8; total * 4];
        self.write_pixels(&mut bgra, 4, 2, 1, 0, Some(3));
        bgra
    }

    /// Convert YCbCr to BGR with conformance window cropping.
    pub fn to_bgr(&self) -> Vec<u8> {
        let total = (self.cropped_width() * self.cropped_height()) as usize;
        let mut bgr = vec![0u8; total * 3];
        self.write_pixels(&mut bgr, 3, 2, 1, 0, None);
        bgr
    }

    /// Write pixels into a pre-allocated RGB buffer. Returns the number of bytes
    /// needed; only writes when `output` is large enough.
    pub fn write_rgb_into(&self, output: &mut [u8]) -> usize {
        let needed = (self.cropped_width() * self.cropped_height()) as usize * 3;
        if output.len() >= needed {
            self.write_pixels(output, 3, 0, 1, 2, None);
        }
        needed
    }

    /// Write pixels into a pre-allocated RGBA buffer. Returns the number of bytes
    /// needed; only writes when `output` is large enough.
    pub fn write_rgba_into(&self, output: &mut [u8]) -> usize {
        let needed = (self.cropped_width() * self.cropped_height()) as usize * 4;
        if output.len() >= needed {
            self.write_pixels(output, 4, 0, 1, 2, Some(3));
        }
        needed
    }

    /// Write pixels into a pre-allocated BGRA buffer. Returns the number of bytes
    /// needed; only writes when `output` is large enough.
    pub fn write_bgra_into(&self, output: &mut [u8]) -> usize {
        let needed = (self.cropped_width() * self.cropped_height()) as usize * 4;
        if output.len() >= needed {
            self.write_pixels(output, 4, 2, 1, 0, Some(3));
        }
        needed
    }

    /// Write pixels into a pre-allocated BGR buffer. Returns the number of bytes
    /// needed; only writes when `output` is large enough.
    pub fn write_bgr_into(&self, output: &mut [u8]) -> usize {
        let needed = (self.cropped_width() * self.cropped_height()) as usize * 3;
        if output.len() >= needed {
            self.write_pixels(output, 3, 2, 1, 0, None);
        }
        needed
    }

    /// Convert YCbCr to RGBA with conformance window cropping.
    /// Uses real alpha values from `alpha_plane` if present, otherwise alpha=255.
    pub fn to_rgba(&self) -> Vec<u8> {
        let total = (self.cropped_width() * self.cropped_height()) as usize;
        let mut rgba = vec![0u8; total * 4];
        self.write_pixels(&mut rgba, 4, 0, 1, 2, Some(3));
        rgba
    }

    /// Get chroma values for a pixel position
    fn get_chroma(&self, x: u32, y: u32, shift: u8) -> (i32, i32) {
        match self.chroma_format {
            0 => (128, 128), // Monochrome - neutral chroma
            1 => {
                // 4:2:0 - both dimensions halved
                let cx = x / 2;
                let cy = y / 2;
                let c_stride = self.c_stride();
                let c_idx = (cy as usize) * c_stride + (cx as usize);
                let cb = if c_idx < self.cb_plane.len() {
                    (self.cb_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                let cr = if c_idx < self.cr_plane.len() {
                    (self.cr_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                (cb, cr)
            }
            2 => {
                // 4:2:2 - horizontal halved
                let cx = x / 2;
                let c_stride = self.c_stride();
                let c_idx = (y as usize) * c_stride + (cx as usize);
                let cb = if c_idx < self.cb_plane.len() {
                    (self.cb_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                let cr = if c_idx < self.cr_plane.len() {
                    (self.cr_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                (cb, cr)
            }
            3 => {
                // 4:4:4 - full resolution
                let c_idx = (y * self.width + x) as usize;
                let cb = if c_idx < self.cb_plane.len() {
                    (self.cb_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                let cr = if c_idx < self.cr_plane.len() {
                    (self.cr_plane[c_idx] >> shift) as i32
                } else {
                    128
                };
                (cb, cr)
            }
            _ => (128, 128),
        }
    }

    /// Set a luma sample
    #[inline]
    pub fn set_y(&mut self, x: u32, y: u32, value: u16) {
        let idx = (y * self.width + x) as usize;
        if idx < self.y_plane.len() {
            self.y_plane[idx] = value;
        }
    }

    /// Set a Cb chroma sample
    #[inline]
    pub fn set_cb(&mut self, x: u32, y: u32, value: u16) {
        let stride = self.c_stride();
        let idx = (y as usize) * stride + (x as usize);
        if idx < self.cb_plane.len() {
            self.cb_plane[idx] = value;
        }
    }

    /// Set a Cr chroma sample
    #[inline]
    pub fn set_cr(&mut self, x: u32, y: u32, value: u16) {
        let stride = self.c_stride();
        let idx = (y as usize) * stride + (x as usize);
        if idx < self.cr_plane.len() {
            self.cr_plane[idx] = value;
        }
    }

    /// Get a luma sample
    #[inline]
    pub fn get_y(&self, x: u32, y: u32) -> u16 {
        let idx = (y * self.width + x) as usize;
        if idx < self.y_plane.len() {
            self.y_plane[idx]
        } else {
            0
        }
    }

    /// Get a Cb chroma sample
    #[inline]
    pub fn get_cb(&self, x: u32, y: u32) -> u16 {
        let stride = self.c_stride();
        let idx = (y as usize) * stride + (x as usize);
        if idx < self.cb_plane.len() {
            self.cb_plane[idx]
        } else {
            128 << (self.bit_depth - 8)
        }
    }

    /// Get a Cr chroma sample
    #[inline]
    pub fn get_cr(&self, x: u32, y: u32) -> u16 {
        let stride = self.c_stride();
        let idx = (y as usize) * stride + (x as usize);
        if idx < self.cr_plane.len() {
            self.cr_plane[idx]
        } else {
            128 << (self.bit_depth - 8)
        }
    }

    /// Get a mutable plane slice and stride for a given component.
    ///
    /// Returns `(plane, stride)` where `plane` is the raw pixel data
    /// and `stride` is the number of pixels per row.
    #[inline]
    pub fn plane_mut(&mut self, c_idx: u8) -> (&mut [u16], usize) {
        match c_idx {
            0 => (&mut self.y_plane, self.width as usize),
            1 => {
                let stride = self.c_stride();
                (&mut self.cb_plane, stride)
            }
            2 => {
                let stride = self.c_stride();
                (&mut self.cr_plane, stride)
            }
            _ => (&mut self.y_plane, self.width as usize),
        }
    }

    /// Get an immutable plane slice and stride for a given component.
    #[inline]
    pub fn plane(&self, c_idx: u8) -> (&[u16], usize) {
        match c_idx {
            0 => (&self.y_plane, self.width as usize),
            1 => {
                let stride = self.c_stride();
                (&self.cb_plane, stride)
            }
            2 => {
                let stride = self.c_stride();
                (&self.cr_plane, stride)
            }
            _ => (&self.y_plane, self.width as usize),
        }
    }

    /// Get chroma plane dimensions (width, height)
    fn chroma_dims(&self) -> (u32, u32) {
        match self.chroma_format {
            0 => (0, 0),
            1 => (self.width.div_ceil(2), self.height.div_ceil(2)),
            2 => (self.width.div_ceil(2), self.height),
            3 => (self.width, self.height),
            _ => (self.width.div_ceil(2), self.height.div_ceil(2)),
        }
    }

    /// Rotate the frame 90° clockwise, returning a new frame
    pub fn rotate_90_cw(&self) -> Self {
        let ow = self.width;
        let oh = self.height;
        let nw = oh;
        let nh = ow;

        // Rotate luma: dst(dx, dy) = src(dy, oh-1-dx)
        let mut y_plane = vec![0u16; (nw * nh) as usize];
        for dy in 0..nh {
            for dx in 0..nw {
                y_plane[(dy * nw + dx) as usize] = self.y_plane[((oh - 1 - dx) * ow + dy) as usize];
            }
        }

        // Rotate alpha plane (same transform as luma)
        let alpha_plane = self.alpha_plane.as_ref().map(|alpha| {
            let mut rotated = vec![0u16; (nw * nh) as usize];
            for dy in 0..nh {
                for dx in 0..nw {
                    rotated[(dy * nw + dx) as usize] = alpha[((oh - 1 - dx) * ow + dy) as usize];
                }
            }
            rotated
        });

        // Rotate chroma planes
        let (ocw, och) = self.chroma_dims();
        if ocw > 0 && och > 0 {
            let ncw = och;
            let nch = ocw;
            let csz = (ncw * nch) as usize;
            let mut cb_plane = vec![0u16; csz];
            let mut cr_plane = vec![0u16; csz];
            for dy in 0..nch {
                for dx in 0..ncw {
                    let si = (och - 1 - dx) as usize * ocw as usize + dy as usize;
                    let di = dy as usize * ncw as usize + dx as usize;
                    if si < self.cb_plane.len() {
                        cb_plane[di] = self.cb_plane[si];
                        cr_plane[di] = self.cr_plane[si];
                    }
                }
            }

            Self {
                width: nw,
                height: nh,
                y_plane,
                cb_plane,
                cr_plane,
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_bottom,
                crop_right: self.crop_top,
                crop_top: self.crop_left,
                crop_bottom: self.crop_right,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        } else {
            Self {
                width: nw,
                height: nh,
                y_plane,
                cb_plane: Vec::new(),
                cr_plane: Vec::new(),
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_bottom,
                crop_right: self.crop_top,
                crop_top: self.crop_left,
                crop_bottom: self.crop_right,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        }
    }

    /// Rotate the frame 180°, returning a new frame
    pub fn rotate_180(&self) -> Self {
        let w = self.width;
        let h = self.height;

        // Rotate luma: dst(dx, dy) = src(w-1-dx, h-1-dy)
        let mut y_plane = vec![0u16; (w * h) as usize];
        for dy in 0..h {
            for dx in 0..w {
                y_plane[(dy * w + dx) as usize] =
                    self.y_plane[((h - 1 - dy) * w + (w - 1 - dx)) as usize];
            }
        }

        // Rotate alpha plane
        let alpha_plane = self.alpha_plane.as_ref().map(|alpha| {
            let mut rotated = vec![0u16; (w * h) as usize];
            for dy in 0..h {
                for dx in 0..w {
                    rotated[(dy * w + dx) as usize] =
                        alpha[((h - 1 - dy) * w + (w - 1 - dx)) as usize];
                }
            }
            rotated
        });

        // Rotate chroma planes
        let (cw, ch) = self.chroma_dims();
        if cw > 0 && ch > 0 {
            let csz = (cw * ch) as usize;
            let mut cb_plane = vec![0u16; csz];
            let mut cr_plane = vec![0u16; csz];
            for dy in 0..ch {
                for dx in 0..cw {
                    let si = (ch - 1 - dy) as usize * cw as usize + (cw - 1 - dx) as usize;
                    let di = dy as usize * cw as usize + dx as usize;
                    if si < self.cb_plane.len() {
                        cb_plane[di] = self.cb_plane[si];
                        cr_plane[di] = self.cr_plane[si];
                    }
                }
            }

            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane,
                cr_plane,
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_right,
                crop_right: self.crop_left,
                crop_top: self.crop_bottom,
                crop_bottom: self.crop_top,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        } else {
            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane: Vec::new(),
                cr_plane: Vec::new(),
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_right,
                crop_right: self.crop_left,
                crop_top: self.crop_bottom,
                crop_bottom: self.crop_top,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        }
    }

    /// Rotate the frame 270° clockwise (= 90° counter-clockwise), returning a new frame
    pub fn rotate_270_cw(&self) -> Self {
        let ow = self.width;
        let oh = self.height;
        let nw = oh;
        let nh = ow;

        // Rotate luma: dst(dx, dy) = src(ow-1-dy, dx)
        let mut y_plane = vec![0u16; (nw * nh) as usize];
        for dy in 0..nh {
            for dx in 0..nw {
                y_plane[(dy * nw + dx) as usize] = self.y_plane[(dx * ow + (ow - 1 - dy)) as usize];
            }
        }

        // Rotate alpha plane
        let alpha_plane = self.alpha_plane.as_ref().map(|alpha| {
            let mut rotated = vec![0u16; (nw * nh) as usize];
            for dy in 0..nh {
                for dx in 0..nw {
                    rotated[(dy * nw + dx) as usize] = alpha[(dx * ow + (ow - 1 - dy)) as usize];
                }
            }
            rotated
        });

        // Rotate chroma planes
        let (ocw, och) = self.chroma_dims();
        if ocw > 0 && och > 0 {
            let ncw = och;
            let nch = ocw;
            let csz = (ncw * nch) as usize;
            let mut cb_plane = vec![0u16; csz];
            let mut cr_plane = vec![0u16; csz];
            for dy in 0..nch {
                for dx in 0..ncw {
                    let si = dx as usize * ocw as usize + (ocw - 1 - dy) as usize;
                    let di = dy as usize * ncw as usize + dx as usize;
                    if si < self.cb_plane.len() {
                        cb_plane[di] = self.cb_plane[si];
                        cr_plane[di] = self.cr_plane[si];
                    }
                }
            }

            Self {
                width: nw,
                height: nh,
                y_plane,
                cb_plane,
                cr_plane,
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_top,
                crop_right: self.crop_bottom,
                crop_top: self.crop_right,
                crop_bottom: self.crop_left,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        } else {
            Self {
                width: nw,
                height: nh,
                y_plane,
                cb_plane: Vec::new(),
                cr_plane: Vec::new(),
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_top,
                crop_right: self.crop_bottom,
                crop_top: self.crop_right,
                crop_bottom: self.crop_left,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        }
    }

    /// Mirror the frame about the vertical axis (left-right flip)
    pub fn mirror_horizontal(&self) -> Self {
        let w = self.width;
        let h = self.height;

        let mut y_plane = vec![0u16; (w * h) as usize];
        for dy in 0..h {
            for dx in 0..w {
                y_plane[(dy * w + dx) as usize] = self.y_plane[(dy * w + (w - 1 - dx)) as usize];
            }
        }

        let alpha_plane = self.alpha_plane.as_ref().map(|alpha| {
            let mut mirrored = vec![0u16; (w * h) as usize];
            for dy in 0..h {
                for dx in 0..w {
                    mirrored[(dy * w + dx) as usize] = alpha[(dy * w + (w - 1 - dx)) as usize];
                }
            }
            mirrored
        });

        let (cw, ch) = self.chroma_dims();
        if cw > 0 && ch > 0 {
            let csz = (cw * ch) as usize;
            let mut cb_plane = vec![0u16; csz];
            let mut cr_plane = vec![0u16; csz];
            for dy in 0..ch {
                for dx in 0..cw {
                    let si = dy as usize * cw as usize + (cw - 1 - dx) as usize;
                    let di = dy as usize * cw as usize + dx as usize;
                    if si < self.cb_plane.len() {
                        cb_plane[di] = self.cb_plane[si];
                        cr_plane[di] = self.cr_plane[si];
                    }
                }
            }
            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane,
                cr_plane,
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_right,
                crop_right: self.crop_left,
                crop_top: self.crop_top,
                crop_bottom: self.crop_bottom,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        } else {
            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane: Vec::new(),
                cr_plane: Vec::new(),
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_right,
                crop_right: self.crop_left,
                crop_top: self.crop_top,
                crop_bottom: self.crop_bottom,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        }
    }

    /// Mirror the frame about the horizontal axis (top-bottom flip)
    pub fn mirror_vertical(&self) -> Self {
        let w = self.width;
        let h = self.height;

        let mut y_plane = vec![0u16; (w * h) as usize];
        for dy in 0..h {
            for dx in 0..w {
                y_plane[(dy * w + dx) as usize] = self.y_plane[((h - 1 - dy) * w + dx) as usize];
            }
        }

        let alpha_plane = self.alpha_plane.as_ref().map(|alpha| {
            let mut mirrored = vec![0u16; (w * h) as usize];
            for dy in 0..h {
                for dx in 0..w {
                    mirrored[(dy * w + dx) as usize] = alpha[((h - 1 - dy) * w + dx) as usize];
                }
            }
            mirrored
        });

        let (cw, ch) = self.chroma_dims();
        if cw > 0 && ch > 0 {
            let csz = (cw * ch) as usize;
            let mut cb_plane = vec![0u16; csz];
            let mut cr_plane = vec![0u16; csz];
            for dy in 0..ch {
                for dx in 0..cw {
                    let si = (ch - 1 - dy) as usize * cw as usize + dx as usize;
                    let di = dy as usize * cw as usize + dx as usize;
                    if si < self.cb_plane.len() {
                        cb_plane[di] = self.cb_plane[si];
                        cr_plane[di] = self.cr_plane[si];
                    }
                }
            }
            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane,
                cr_plane,
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_left,
                crop_right: self.crop_right,
                crop_top: self.crop_bottom,
                crop_bottom: self.crop_top,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        } else {
            Self {
                width: w,
                height: h,
                y_plane,
                cb_plane: Vec::new(),
                cr_plane: Vec::new(),
                bit_depth: self.bit_depth,
                chroma_format: self.chroma_format,
                crop_left: self.crop_left,
                crop_right: self.crop_right,
                crop_top: self.crop_bottom,
                crop_bottom: self.crop_top,
                deblock_flags: Vec::new(),
                deblock_stride: 0,
                qp_map: Vec::new(),
                alpha_plane,
                full_range: self.full_range,
                matrix_coeffs: self.matrix_coeffs,
            }
        }
    }
}
