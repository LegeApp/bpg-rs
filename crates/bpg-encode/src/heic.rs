//! HEIF/HEIC still-image writer.
//!
//! HEVC parameter sets are carried in `hvcC`; image item data contains the
//! remaining NAL units with four-byte length prefixes.  The writer deliberately
//! implements the compact, item-based still-image subset rather than the timed
//! track machinery used for video.

use bpg_image::{ChromaFormat, ColorSpace, Image};

use crate::{encode_annexb_still, EncodeError, EncodedHevcStill, EncoderTuning, HevcEncoder};

const PRIMARY_ITEM_ID: u16 = 1;

/// Display orientation copied from source metadata and represented with HEIF
/// transformative properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageOrientation {
    #[default]
    Normal,
    MirrorHorizontal,
    Rotate180,
    MirrorVertical,
    Transpose,
    Rotate90,
    Transverse,
    Rotate270,
}

/// Optional data associated with a HEIC primary image.
#[derive(Debug, Default)]
pub struct HeicEncodeOptions {
    /// An already color-converted thumbnail. The library does not resize the
    /// primary implicitly; the CLI supplies its standard 320-pixel thumbnail.
    pub thumbnail: Option<Image>,
    /// TIFF payload, without a JPEG `Exif\0\0` marker or HEIF offset prefix.
    pub exif: Option<Vec<u8>>,
    /// A complete UTF-8 XMP packet.
    pub xmp: Option<Vec<u8>>,
    pub orientation: ImageOrientation,
}

/// Encode one image and optional associated items to a complete HEIC file.
pub fn encode_heic_still_image(
    image: Image,
    backend: &dyn HevcEncoder,
    qp: u8,
    compress_level: u8,
    tuning: EncoderTuning,
    mut options: HeicEncodeOptions,
) -> Result<Vec<u8>, EncodeError> {
    let main = encode_annexb_still(image, backend, qp, compress_level, tuning.clone())?;
    let thumbnail = options
        .thumbnail
        .take()
        .map(|thumb| encode_annexb_still(thumb, backend, qp, compress_level, tuning))
        .transpose()?;
    package_heic(main, thumbnail, options)
}

#[derive(Debug, Clone)]
struct NalUnit {
    nal_type: u8,
    data: Vec<u8>,
}

#[derive(Debug)]
struct CodedItem {
    encoded: EncodedHevcStill,
    hvcc: Vec<u8>,
    payload: Vec<u8>,
    profile_idc: u8,
}

#[derive(Debug)]
struct ItemRecord {
    id: u16,
    item_type: [u8; 4],
    name: &'static str,
    content_type: Option<&'static str>,
    data: Vec<u8>,
}

#[derive(Debug)]
struct Association {
    item_id: u16,
    properties: Vec<(u8, bool)>,
}

#[derive(Debug)]
struct ItemReference {
    reference_type: [u8; 4],
    from_id: u16,
    to_id: u16,
}

fn package_heic(
    main: EncodedHevcStill,
    thumbnail: Option<EncodedHevcStill>,
    options: HeicEncodeOptions,
) -> Result<Vec<u8>, EncodeError> {
    let main = prepare_coded_item(main)?;
    let thumbnail = thumbnail.map(prepare_coded_item).transpose()?;

    let major_brand = if matches!(main.profile_idc, 1 | 3) {
        *b"heic"
    } else {
        *b"heix"
    };
    let ftyp = file_type_box(major_brand)?;

    let mut items = Vec::new();
    items.push(ItemRecord {
        id: PRIMARY_ITEM_ID,
        item_type: *b"hvc1",
        name: "Primary",
        content_type: None,
        data: main.payload.clone(),
    });

    let mut properties = Vec::<Vec<u8>>::new();
    let mut associations = Vec::<Association>::new();
    let main_props = append_coded_properties(&mut properties, &main, options.orientation)?;
    associations.push(Association {
        item_id: PRIMARY_ITEM_ID,
        properties: main_props,
    });

    let mut references = Vec::new();
    let mut next_id = PRIMARY_ITEM_ID + 1;
    if let Some(thumb) = &thumbnail {
        let thumb_id = next_id;
        next_id += 1;
        items.push(ItemRecord {
            id: thumb_id,
            item_type: *b"hvc1",
            name: "Thumbnail",
            content_type: None,
            data: thumb.payload.clone(),
        });
        let thumb_props = append_coded_properties(&mut properties, thumb, options.orientation)?;
        associations.push(Association {
            item_id: thumb_id,
            properties: thumb_props,
        });
        references.push(ItemReference {
            reference_type: *b"thmb",
            from_id: thumb_id,
            to_id: PRIMARY_ITEM_ID,
        });
    }

    if let Some(exif) = options.exif {
        let id = next_id;
        next_id += 1;
        let mut data = Vec::with_capacity(exif.len() + 4);
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&exif);
        items.push(ItemRecord {
            id,
            item_type: *b"Exif",
            name: "Exif",
            content_type: None,
            data,
        });
        references.push(ItemReference {
            reference_type: *b"cdsc",
            from_id: id,
            to_id: PRIMARY_ITEM_ID,
        });
    }
    if let Some(xmp) = options.xmp {
        let id = next_id;
        items.push(ItemRecord {
            id,
            item_type: *b"mime",
            name: "XMP",
            content_type: Some("application/rdf+xml"),
            data: xmp,
        });
        references.push(ItemReference {
            reference_type: *b"cdsc",
            from_id: id,
            to_id: PRIMARY_ITEM_ID,
        });
    }

    let zero_offsets = vec![0u32; items.len()];
    let meta0 = meta_box(
        &items,
        &properties,
        &associations,
        &references,
        &zero_offsets,
    )?;
    let mdat_content_offset = checked_u32(ftyp.len() + meta0.len() + 8, "mdat offset")?;
    let mut offsets = Vec::with_capacity(items.len());
    let mut cursor = mdat_content_offset;
    for item in &items {
        offsets.push(cursor);
        cursor = cursor
            .checked_add(checked_u32(item.data.len(), "item length")?)
            .ok_or_else(|| EncodeError::Container("HEIC exceeds 4 GiB".into()))?;
    }
    let meta = meta_box(&items, &properties, &associations, &references, &offsets)?;
    if meta.len() != meta0.len() {
        return Err(EncodeError::Container(
            "internal error: iloc patch changed meta size".into(),
        ));
    }

    let mut mdat_payload = Vec::new();
    for item in &items {
        mdat_payload.extend_from_slice(&item.data);
    }
    let mdat = make_box(*b"mdat", mdat_payload)?;
    let total = ftyp
        .len()
        .checked_add(meta.len())
        .and_then(|n| n.checked_add(mdat.len()))
        .ok_or_else(|| EncodeError::Container("HEIC size overflow".into()))?;
    checked_u32(total, "HEIC file size")?;

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&meta);
    out.extend_from_slice(&mdat);
    Ok(out)
}

fn prepare_coded_item(encoded: EncodedHevcStill) -> Result<CodedItem, EncodeError> {
    let nals = split_annexb(&encoded.annexb)?;
    let vps = exactly_one(&nals, 32, "VPS")?;
    let sps = exactly_one(&nals, 33, "SPS")?;
    let pps = exactly_one(&nals, 34, "PPS")?;
    let (hvcc, profile_idc) = build_hvcc(vps, sps, pps, &encoded)?;

    let mut payload = Vec::new();
    for nal in nals
        .iter()
        .filter(|nal| !matches!(nal.nal_type, 32 | 33 | 34))
    {
        let len = checked_u32(nal.data.len(), "NAL length")?;
        payload.extend_from_slice(&len.to_be_bytes());
        payload.extend_from_slice(&nal.data);
    }
    if payload.is_empty() {
        return Err(EncodeError::Container(
            "HEVC stream has no image NAL".into(),
        ));
    }
    Ok(CodedItem {
        encoded,
        hvcc,
        payload,
        profile_idc,
    })
}

fn exactly_one<'a>(nals: &'a [NalUnit], nal_type: u8, name: &str) -> Result<&'a [u8], EncodeError> {
    let mut found = nals.iter().filter(|n| n.nal_type == nal_type);
    let first = found
        .next()
        .ok_or_else(|| EncodeError::Container(format!("HEVC stream is missing {name}")))?;
    if found.next().is_some() {
        return Err(EncodeError::Container(format!(
            "HEVC stream contains multiple {name} NALs"
        )));
    }
    Ok(&first.data)
}

fn split_annexb(data: &[u8]) -> Result<Vec<NalUnit>, EncodeError> {
    fn start_code(data: &[u8], at: usize) -> Option<usize> {
        if data.get(at..at + 4) == Some(&[0, 0, 0, 1]) {
            Some(4)
        } else if data.get(at..at + 3) == Some(&[0, 0, 1]) {
            Some(3)
        } else {
            None
        }
    }

    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if let Some(len) = start_code(data, i) {
            starts.push((i, len));
            i += len;
        } else {
            i += 1;
        }
    }
    if starts.is_empty() || starts[0].0 != 0 {
        return Err(EncodeError::Container(
            "HEVC backend did not return Annex-B".into(),
        ));
    }
    let mut out = Vec::with_capacity(starts.len());
    for (idx, &(start, prefix)) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).map_or(data.len(), |s| s.0);
        let raw = &data[start + prefix..end];
        if raw.len() < 2 {
            return Err(EncodeError::Container("truncated HEVC NAL".into()));
        }
        out.push(NalUnit {
            nal_type: (raw[0] >> 1) & 0x3f,
            data: raw.to_vec(),
        });
    }
    Ok(out)
}

fn unescape_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    for &byte in ebsp {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        rbsp.push(byte);
        if byte == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    rbsp
}

fn build_hvcc(
    vps: &[u8],
    sps: &[u8],
    pps: &[u8],
    encoded: &EncodedHevcStill,
) -> Result<(Vec<u8>, u8), EncodeError> {
    if sps.len() < 3 {
        return Err(EncodeError::Container("SPS is too short".into()));
    }
    let rbsp = unescape_rbsp(&sps[2..]);
    // The first SPS payload byte contains vps_id, max_sub_layers and nesting;
    // the following 12 bytes are general_profile_tier_level for one sublayer.
    if rbsp.len() < 13 {
        return Err(EncodeError::Container(
            "SPS profile-tier-level is truncated".into(),
        ));
    }
    let ptl = &rbsp[1..13];
    let profile_idc = ptl[0] & 0x1f;
    let chroma = chroma_format_idc(encoded.chroma_format);
    let depth_minus8 = encoded.bit_depth - 8;

    let mut out = Vec::new();
    out.push(1); // configurationVersion
    out.extend_from_slice(ptl); // profile byte, compatibility, constraints, level
    out.extend_from_slice(&0xf000u16.to_be_bytes()); // min_spatial_segmentation_idc
    out.push(0xfc); // parallelismType = unknown
    out.push(0xfc | chroma);
    out.push(0xf8 | depth_minus8);
    out.push(0xf8 | depth_minus8);
    out.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate
    out.push(0x0f); // 1 temporal layer, nested, four-byte NAL lengths
    out.push(3); // VPS, SPS, PPS arrays
    for (nal_type, nal) in [(32u8, vps), (33, sps), (34, pps)] {
        let len = u16::try_from(nal.len())
            .map_err(|_| EncodeError::Container("parameter-set NAL exceeds 64 KiB".into()))?;
        out.push(0x80 | nal_type); // array_completeness
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nal);
    }
    Ok((out, profile_idc))
}

fn append_coded_properties(
    all: &mut Vec<Vec<u8>>,
    item: &CodedItem,
    orientation: ImageOrientation,
) -> Result<Vec<(u8, bool)>, EncodeError> {
    let mut assoc = Vec::new();
    push_property(
        all,
        &mut assoc,
        make_box(*b"hvcC", item.hvcc.clone())?,
        true,
    )?;

    let mut ispe = full_box_header(0, 0);
    ispe.extend_from_slice(&item.encoded.coded_width.to_be_bytes());
    ispe.extend_from_slice(&item.encoded.coded_height.to_be_bytes());
    push_property(all, &mut assoc, make_box(*b"ispe", ispe)?, true)?;

    let channels = if item.encoded.chroma_format == ChromaFormat::Gray {
        1
    } else {
        3
    };
    let mut pixi = full_box_header(0, 0);
    pixi.push(channels);
    pixi.extend(std::iter::repeat(item.encoded.bit_depth).take(channels as usize));
    push_property(all, &mut assoc, make_box(*b"pixi", pixi)?, false)?;

    let (primaries, transfer, matrix) =
        nclx_codes(item.encoded.color_space, item.encoded.bit_depth);
    let mut colr = Vec::with_capacity(11);
    colr.extend_from_slice(b"nclx");
    colr.extend_from_slice(&primaries.to_be_bytes());
    colr.extend_from_slice(&transfer.to_be_bytes());
    colr.extend_from_slice(&matrix.to_be_bytes());
    colr.push(if item.encoded.limited_range { 0 } else { 0x80 });
    push_property(all, &mut assoc, make_box(*b"colr", colr)?, false)?;

    if item.encoded.display_width != item.encoded.coded_width
        || item.encoded.display_height != item.encoded.coded_height
    {
        let mut clap = Vec::with_capacity(32);
        clap.extend_from_slice(&item.encoded.display_width.to_be_bytes());
        clap.extend_from_slice(&1u32.to_be_bytes());
        clap.extend_from_slice(&item.encoded.display_height.to_be_bytes());
        clap.extend_from_slice(&1u32.to_be_bytes());
        let h_off = item.encoded.display_width as i64 - item.encoded.coded_width as i64;
        let v_off = item.encoded.display_height as i64 - item.encoded.coded_height as i64;
        clap.extend_from_slice(&(h_off as i32).to_be_bytes());
        clap.extend_from_slice(&2u32.to_be_bytes());
        clap.extend_from_slice(&(v_off as i32).to_be_bytes());
        clap.extend_from_slice(&2u32.to_be_bytes());
        push_property(all, &mut assoc, make_box(*b"clap", clap)?, true)?;
    }

    for property in orientation_properties(orientation)? {
        push_property(all, &mut assoc, property, true)?;
    }
    Ok(assoc)
}

fn push_property(
    all: &mut Vec<Vec<u8>>,
    assoc: &mut Vec<(u8, bool)>,
    property: Vec<u8>,
    essential: bool,
) -> Result<(), EncodeError> {
    all.push(property);
    let index = u8::try_from(all.len())
        .map_err(|_| EncodeError::Container("too many HEIF properties".into()))?;
    if index > 0x7f {
        return Err(EncodeError::Container("too many HEIF properties".into()));
    }
    assoc.push((index, essential));
    Ok(())
}

fn orientation_properties(orientation: ImageOrientation) -> Result<Vec<Vec<u8>>, EncodeError> {
    let irot = |value: u8| make_box(*b"irot", vec![value & 3]);
    let imir = |axis: u8| make_box(*b"imir", vec![axis & 1]);
    match orientation {
        ImageOrientation::Normal => Ok(Vec::new()),
        ImageOrientation::MirrorHorizontal => Ok(vec![imir(0)?]),
        ImageOrientation::Rotate180 => Ok(vec![irot(2)?]),
        ImageOrientation::MirrorVertical => Ok(vec![imir(1)?]),
        ImageOrientation::Transpose => Ok(vec![imir(0)?, irot(3)?]),
        ImageOrientation::Rotate90 => Ok(vec![irot(3)?]),
        ImageOrientation::Transverse => Ok(vec![imir(0)?, irot(1)?]),
        ImageOrientation::Rotate270 => Ok(vec![irot(1)?]),
    }
}

fn nclx_codes(color: ColorSpace, bit_depth: u8) -> (u16, u16, u16) {
    match color {
        ColorSpace::YCbCr => (6, 6, 6),
        ColorSpace::YCbCrBt709 => (1, 1, 1),
        ColorSpace::YCbCrBt2020 => (9, if bit_depth <= 10 { 14 } else { 15 }, 9),
        ColorSpace::Rgb => (1, 13, 0),
        ColorSpace::YCgCo => (1, 13, 8),
    }
}

fn chroma_format_idc(chroma: ChromaFormat) -> u8 {
    match chroma {
        ChromaFormat::Gray => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

fn file_type_box(major: [u8; 4]) -> Result<Vec<u8>, EncodeError> {
    let mut payload = Vec::with_capacity(20);
    payload.extend_from_slice(&major);
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(b"mif1");
    payload.extend_from_slice(&major);
    make_box(*b"ftyp", payload)
}

fn meta_box(
    items: &[ItemRecord],
    properties: &[Vec<u8>],
    associations: &[Association],
    references: &[ItemReference],
    offsets: &[u32],
) -> Result<Vec<u8>, EncodeError> {
    let mut payload = full_box_header(0, 0);
    payload.extend_from_slice(&handler_box()?);
    payload.extend_from_slice(&primary_item_box()?);
    payload.extend_from_slice(&item_location_box(items, offsets)?);
    payload.extend_from_slice(&item_info_box(items)?);
    if !references.is_empty() {
        payload.extend_from_slice(&item_reference_box(references)?);
    }
    payload.extend_from_slice(&item_properties_box(properties, associations)?);
    make_box(*b"meta", payload)
}

fn handler_box() -> Result<Vec<u8>, EncodeError> {
    let mut payload = full_box_header(0, 0);
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(b"pict");
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(b"bpg-rs\0");
    make_box(*b"hdlr", payload)
}

fn primary_item_box() -> Result<Vec<u8>, EncodeError> {
    let mut payload = full_box_header(0, 0);
    payload.extend_from_slice(&PRIMARY_ITEM_ID.to_be_bytes());
    make_box(*b"pitm", payload)
}

fn item_location_box(items: &[ItemRecord], offsets: &[u32]) -> Result<Vec<u8>, EncodeError> {
    if items.len() != offsets.len() {
        return Err(EncodeError::Container("iloc item/offset mismatch".into()));
    }
    let count = u16::try_from(items.len())
        .map_err(|_| EncodeError::Container("too many HEIF items".into()))?;
    let mut payload = full_box_header(0, 0);
    payload.push(0x44); // four-byte extent offset and length
    payload.push(0x00); // no base offset
    payload.extend_from_slice(&count.to_be_bytes());
    for (item, &offset) in items.iter().zip(offsets) {
        payload.extend_from_slice(&item.id.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index
        payload.extend_from_slice(&1u16.to_be_bytes()); // extent_count
        payload.extend_from_slice(&offset.to_be_bytes());
        payload.extend_from_slice(&checked_u32(item.data.len(), "item length")?.to_be_bytes());
    }
    make_box(*b"iloc", payload)
}

fn item_info_box(items: &[ItemRecord]) -> Result<Vec<u8>, EncodeError> {
    let count = u16::try_from(items.len())
        .map_err(|_| EncodeError::Container("too many HEIF items".into()))?;
    let mut payload = full_box_header(0, 0);
    payload.extend_from_slice(&count.to_be_bytes());
    for item in items {
        let mut infe = full_box_header(2, 0);
        infe.extend_from_slice(&item.id.to_be_bytes());
        infe.extend_from_slice(&0u16.to_be_bytes());
        infe.extend_from_slice(&item.item_type);
        infe.extend_from_slice(item.name.as_bytes());
        infe.push(0);
        if item.item_type == *b"mime" {
            infe.extend_from_slice(
                item.content_type
                    .unwrap_or("application/octet-stream")
                    .as_bytes(),
            );
            infe.push(0);
            infe.push(0); // content_encoding
        }
        payload.extend_from_slice(&make_box(*b"infe", infe)?);
    }
    make_box(*b"iinf", payload)
}

fn item_reference_box(references: &[ItemReference]) -> Result<Vec<u8>, EncodeError> {
    let mut payload = full_box_header(0, 0);
    for reference in references {
        let mut child = Vec::with_capacity(6);
        child.extend_from_slice(&reference.from_id.to_be_bytes());
        child.extend_from_slice(&1u16.to_be_bytes());
        child.extend_from_slice(&reference.to_id.to_be_bytes());
        payload.extend_from_slice(&make_box(reference.reference_type, child)?);
    }
    make_box(*b"iref", payload)
}

fn item_properties_box(
    properties: &[Vec<u8>],
    associations: &[Association],
) -> Result<Vec<u8>, EncodeError> {
    let mut ipco_payload = Vec::new();
    for property in properties {
        ipco_payload.extend_from_slice(property);
    }
    let ipco = make_box(*b"ipco", ipco_payload)?;

    let mut ipma_payload = full_box_header(0, 0);
    ipma_payload.extend_from_slice(&checked_u32(associations.len(), "ipma count")?.to_be_bytes());
    for assoc in associations {
        ipma_payload.extend_from_slice(&assoc.item_id.to_be_bytes());
        let count = u8::try_from(assoc.properties.len())
            .map_err(|_| EncodeError::Container("too many item properties".into()))?;
        ipma_payload.push(count);
        for &(index, essential) in &assoc.properties {
            ipma_payload.push(index | if essential { 0x80 } else { 0 });
        }
    }
    let ipma = make_box(*b"ipma", ipma_payload)?;

    let mut payload = Vec::with_capacity(ipco.len() + ipma.len());
    payload.extend_from_slice(&ipco);
    payload.extend_from_slice(&ipma);
    make_box(*b"iprp", payload)
}

fn full_box_header(version: u8, flags: u32) -> Vec<u8> {
    vec![
        version,
        ((flags >> 16) & 0xff) as u8,
        ((flags >> 8) & 0xff) as u8,
        (flags & 0xff) as u8,
    ]
}

fn make_box(box_type: [u8; 4], payload: Vec<u8>) -> Result<Vec<u8>, EncodeError> {
    let size = payload
        .len()
        .checked_add(8)
        .ok_or_else(|| EncodeError::Container("box size overflow".into()))?;
    let size = checked_u32(size, "box size")?;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(&box_type);
    out.extend_from_slice(&payload);
    Ok(out)
}

fn checked_u32(value: usize, what: &str) -> Result<u32, EncodeError> {
    u32::try_from(value)
        .map_err(|_| EncodeError::Container(format!("{what} exceeds 32-bit HEIF limit")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb_split_preserves_ebsp_nals() {
        let data = [0, 0, 0, 1, 0x40, 1, 0xaa, 0, 0, 1, 0x42, 1, 0, 0, 3, 1];
        let nals = split_annexb(&data).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_type, 32);
        assert_eq!(nals[1].nal_type, 33);
        assert_eq!(nals[1].data, [0x42, 1, 0, 0, 3, 1]);
    }

    #[test]
    fn orientation_property_codes_match_decoder() {
        let rotate = orientation_properties(ImageOrientation::Rotate90).unwrap();
        assert_eq!(&rotate[0][4..8], b"irot");
        assert_eq!(rotate[0][8], 3);
        let mirror = orientation_properties(ImageOrientation::MirrorHorizontal).unwrap();
        assert_eq!(&mirror[0][4..8], b"imir");
        assert_eq!(mirror[0][8], 0);
    }
}
