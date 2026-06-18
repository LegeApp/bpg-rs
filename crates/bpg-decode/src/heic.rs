//! HEIC/HEIF still-image decode path.
//!
//! Ported from the `heic-decoder-rs` library (the same HEVC decoder used by
//! `bpg-hevc-decode`). The HEIF container parser is self-contained here;
//! the actual HEVC bitstream decode is delegated to
//! [`bpg_hevc_decode::hevc`].
//!
//! # Public API
//!
//! * [`get_heic_image_info`] — parse HEIF container metadata only
//! * [`decode_heic_to_frame`] — decode to raw YCbCr [`DecodedFrame`]
//! * [`decode_heic`] — decode to interleaved pixel data (RGB/RGBA/BGR/BGRA)
//! * [`decode_heic_thumbnail`] — decode embedded thumbnail when present
//!
//! All functions return [`DecodeError`](crate::DecodeError).

use std::string::{String, ToString};
use std::vec::Vec;
use std::str;

use bpg_hevc_decode::hevc;
pub use bpg_hevc_decode::DecodedFrame;

use crate::{DecodeError, DecodeOutput, PixelLayout};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metadata extracted from a HEIC/HEIF file without full decode.
#[derive(Debug, Clone, Copy)]
pub struct HeicImageInfo {
    /// Width of the primary image in pixels.
    pub width: u32,
    /// Height of the primary image in pixels.
    pub height: u32,
    /// Whether the primary image has an alpha channel.
    pub has_alpha: bool,
    /// Luma bit depth.
    pub bit_depth: u8,
    /// Chroma format: 0 = mono, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_format: u8,
    /// Whether the file contains an embedded EXIF block.
    pub has_exif: bool,
    /// Whether the file contains embedded XMP metadata.
    pub has_xmp: bool,
    /// Whether the file contains an embedded thumbnail image.
    pub has_thumbnail: bool,
}

// ---------------------------------------------------------------------------
// ISOBMFF / HEIF container types (local — not re-exported)
// ---------------------------------------------------------------------------

/// Four-character code identifying a box type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FourCC(pub [u8; 4]);

impl FourCC {
    const FTYP: Self = Self(*b"ftyp");
    const META: Self = Self(*b"meta");
    const PITM: Self = Self(*b"pitm");
    const ILOC: Self = Self(*b"iloc");
    const IINF: Self = Self(*b"iinf");
    const INFE: Self = Self(*b"infe");
    const IPRP: Self = Self(*b"iprp");
    const IPCO: Self = Self(*b"ipco");
    const IPMA: Self = Self(*b"ipma");
    const ISPE: Self = Self(*b"ispe");
    const HVCC: Self = Self(*b"hvcC");
    const HVCB: Self = Self(*b"hvcB");
    const COLR: Self = Self(*b"colr");
    const CLAP: Self = Self(*b"clap");
    const IROT: Self = Self(*b"irot");
    const IMIR: Self = Self(*b"imir");
    const AUXC: Self = Self(*b"auxC");
    const IREF: Self = Self(*b"iref");
    const IDAT: Self = Self(*b"idat");
    const DIMG: Self = Self(*b"dimg");
    const AUXL: Self = Self(*b"auxl");
    const THMB: Self = Self(*b"thmb");

    fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() >= 4 {
            Some(Self([b[0], b[1], b[2], b[3]]))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Box iterator
// ---------------------------------------------------------------------------

struct BoxIter<'a> {
    data: &'a [u8],
    offset: usize,
}

struct IsobmffBox<'a> {
    box_type: FourCC,
    content: &'a [u8],
    /// Absolute offset of the first content byte within the parent slice.
    content_offset: usize,
}

impl<'a> BoxIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for BoxIter<'a> {
    type Item = IsobmffBox<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + 8 > self.data.len() {
            return None;
        }
        let data = &self.data[self.offset..];
        let size_32 = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let box_type = FourCC::from_bytes(&data[4..8])?;

        let (size, header_size): (u64, usize) = if size_32 == 1 {
            if data.len() < 16 {
                return None;
            }
            let ext = u64::from_be_bytes([
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            ]);
            (ext, 16)
        } else if size_32 == 0 {
            ((self.data.len() - self.offset) as u64, 8)
        } else {
            (size_32 as u64, 8)
        };

        let size_usize = usize::try_from(size).ok()?;
        if size_usize < header_size {
            return None;
        }
        let box_end = self.offset.checked_add(size_usize)?;
        if box_end > self.data.len() {
            return None;
        }

        let content_offset = self.offset + header_size;
        let content = &data[header_size..size_usize];
        self.offset = box_end;
        Some(IsobmffBox { box_type, content, content_offset })
    }
}

// ---------------------------------------------------------------------------
// Parsed container structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ItemLocation {
    item_id: u32,
    construction_method: u8,
    base_offset: u64,
    extents: Vec<(u64, u64)>,
}

#[derive(Debug, Clone)]
struct ItemInfo {
    item_id: u32,
    item_type: FourCC,
    item_name: String,
    content_type: String,
    hidden: bool,
}

/// HEVC decoder configuration from an hvcC box.
/// Carries the full set of hvcC fields so grid/overlay paths can read
/// chroma_format and bit_depth without decoding a tile first.
#[derive(Debug, Clone)]
struct HvcCConfig {
    pub chroma_format: u8,
    pub bit_depth_luma_minus8: u8,
    pub length_size_minus_one: u8,
    pub nal_units: Vec<Vec<u8>>,
}

impl HvcCConfig {
    /// Convert to the minimal `HevcDecoderConfig` used by `bpg_hevc_decode`.
    fn to_bhd(&self) -> bpg_hevc_decode::heif::HevcDecoderConfig {
        bpg_hevc_decode::heif::HevcDecoderConfig {
            length_size_minus_one: self.length_size_minus_one,
            nal_units: self.nal_units.clone(),
        }
    }
}

#[derive(Debug, Clone)]
enum ColorInfoKind {
    IccProfile(Vec<u8>),
    Nclx {
        color_primaries: u16,
        transfer_characteristics: u16,
        matrix_coefficients: u16,
        full_range: bool,
    },
}

#[derive(Debug, Clone)]
struct CleanAperture {
    width_n: u32,
    width_d: u32,
    height_n: u32,
    height_d: u32,
    horiz_off_n: i32,
    horiz_off_d: u32,
    vert_off_n: i32,
    vert_off_d: u32,
}

#[derive(Debug, Clone, Copy)]
struct ImageRotation {
    angle: u16, // degrees CW: 0, 90, 180, 270
}

#[derive(Debug, Clone, Copy)]
struct ImageMirror {
    axis: u8, // 0 = vertical axis (L-R flip), 1 = horizontal axis (T-B flip)
}

#[derive(Debug, Clone)]
enum Transform {
    CleanAperture(CleanAperture),
    Mirror(ImageMirror),
    Rotation(ImageRotation),
}

#[derive(Debug, Clone)]
enum ItemProperty {
    ImageExtents { width: u32, height: u32 },
    HevcConfig(HvcCConfig),
    ColorInfo(ColorInfoKind),
    CleanAperture(CleanAperture),
    Rotation(ImageRotation),
    Mirror(ImageMirror),
    AuxiliaryType(String),
    Unknown,
}

#[derive(Debug, Clone)]
struct PropertyAssoc {
    item_id: u32,
    properties: Vec<(u16, bool)>, // (1-based index, essential)
}

#[derive(Debug, Clone)]
struct ItemReference {
    reference_type: FourCC,
    from_item_id: u32,
    to_item_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Hvc1,
    Grid,
    Iovl,
    Iden,
    Unknown,
}

impl From<FourCC> for ItemKind {
    fn from(f: FourCC) -> Self {
        match &f.0 {
            b"hvc1" => Self::Hvc1,
            b"grid" => Self::Grid,
            b"iovl" => Self::Iovl,
            b"iden" => Self::Iden,
            _ => Self::Unknown,
        }
    }
}

/// Resolved item with all associated properties applied.
#[derive(Debug)]
struct Item {
    id: u32,
    kind: ItemKind,
    dimensions: Option<(u32, u32)>,
    hevc_config: Option<HvcCConfig>,
    transforms: Vec<Transform>,
    color_info: Option<ColorInfoKind>,
    auxiliary_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Parsed HEIF container
// ---------------------------------------------------------------------------

struct HeifContainer<'a> {
    data: &'a [u8],
    primary_item_id: u32,
    item_locations: Vec<ItemLocation>,
    item_infos: Vec<ItemInfo>,
    properties: Vec<ItemProperty>,
    property_assocs: Vec<PropertyAssoc>,
    item_references: Vec<ItemReference>,
    idat_data: Option<&'a [u8]>,
}

impl<'a> HeifContainer<'a> {
    fn primary_item(&self) -> Option<Item> {
        self.get_item(self.primary_item_id)
    }

    fn get_item(&self, item_id: u32) -> Option<Item> {
        let info = self.item_infos.iter().find(|i| i.item_id == item_id)?;
        let assoc = self.property_assocs.iter().find(|a| a.item_id == item_id);

        let mut dimensions = None;
        let mut hevc_config = None;
        let mut transforms = Vec::new();
        let mut color_info = None;
        let mut auxiliary_type = None;

        if let Some(assoc) = assoc {
            for &(prop_idx, _essential) in &assoc.properties {
                if prop_idx == 0 {
                    continue;
                }
                let idx = prop_idx as usize - 1;
                if let Some(prop) = self.properties.get(idx) {
                    match prop {
                        ItemProperty::ImageExtents { width, height } => {
                            dimensions = Some((*width, *height));
                        }
                        ItemProperty::HevcConfig(cfg) => {
                            hevc_config = Some(cfg.clone());
                        }
                        ItemProperty::CleanAperture(clap) => {
                            transforms.push(Transform::CleanAperture(clap.clone()));
                        }
                        ItemProperty::Rotation(rot) => {
                            transforms.push(Transform::Rotation(*rot));
                        }
                        ItemProperty::Mirror(m) => {
                            transforms.push(Transform::Mirror(*m));
                        }
                        ItemProperty::ColorInfo(ci) => {
                            color_info = Some(ci.clone());
                        }
                        ItemProperty::AuxiliaryType(s) => {
                            auxiliary_type = Some(s.clone());
                        }
                        _ => {}
                    }
                }
            }
        }

        Some(Item {
            id: item_id,
            kind: ItemKind::from(info.item_type),
            dimensions,
            hevc_config,
            transforms,
            color_info,
            auxiliary_type,
        })
    }

    fn get_item_data(&self, item_id: u32) -> Option<&'a [u8]> {
        let loc = self.item_locations.iter().find(|l| l.item_id == item_id)?;
        if loc.extents.is_empty() {
            return None;
        }
        let source = match loc.construction_method {
            0 => self.data,
            1 => self.idat_data?,
            _ => return None,
        };
        // Single extent fast path
        if loc.extents.len() == 1 {
            let (off, len) = loc.extents[0];
            let start = usize::try_from(loc.base_offset.checked_add(off)?).ok()?;
            let length = usize::try_from(len).ok()?;
            let end = start.checked_add(length)?;
            if end <= source.len() {
                return Some(&source[start..end]);
            }
            return None;
        }
        // Multiple extents: return first only (uncommon for still images)
        let (off, len) = loc.extents[0];
        let start = usize::try_from(loc.base_offset.checked_add(off)?).ok()?;
        let length = usize::try_from(len).ok()?;
        let end = start.checked_add(length)?;
        if end <= source.len() { Some(&source[start..end]) } else { None }
    }

    fn find_auxiliary_items(&self, target_id: u32, aux_type_prefix: &str) -> Vec<u32> {
        self.item_references
            .iter()
            .filter(|r| r.reference_type == FourCC::AUXL && r.to_item_ids.contains(&target_id))
            .filter_map(|r| {
                let item = self.get_item(r.from_item_id)?;
                if let Some(ref at) = item.auxiliary_type {
                    if at.starts_with(aux_type_prefix) {
                        return Some(r.from_item_id);
                    }
                }
                None
            })
            .collect()
    }

    fn get_item_references(&self, from_id: u32, ref_type: FourCC) -> Vec<u32> {
        self.item_references
            .iter()
            .filter(|r| r.from_item_id == from_id && r.reference_type == ref_type)
            .flat_map(|r| r.to_item_ids.iter().copied())
            .collect()
    }

    fn find_thumbnails(&self, target_id: u32) -> Vec<u32> {
        self.item_references
            .iter()
            .filter(|r| r.reference_type == FourCC::THMB && r.to_item_ids.contains(&target_id))
            .map(|r| r.from_item_id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Container parser
// ---------------------------------------------------------------------------

type R<T> = Result<T, DecodeError>;

fn container_err(msg: &'static str) -> DecodeError {
    DecodeError::Container(msg)
}

fn invalid_data(msg: &'static str) -> DecodeError {
    DecodeError::InvalidData(msg)
}

fn parse_container(data: &[u8]) -> R<HeifContainer<'_>> {
    let mut brand = FourCC(*b"    ");
    let mut compatible_brands: Vec<FourCC> = Vec::new();
    let mut primary_item_id = 0u32;
    let mut item_locations = Vec::new();
    let mut item_infos = Vec::new();
    let mut properties = Vec::new();
    let mut property_assocs = Vec::new();
    let mut item_references = Vec::new();
    let mut idat_data = None;

    for top in BoxIter::new(data) {
        match top.box_type {
            FourCC::FTYP => {
                parse_ftyp(top.content, &mut brand, &mut compatible_brands)?;
            }
            FourCC::META => {
                parse_meta(
                    top.content,
                    data,
                    &mut primary_item_id,
                    &mut item_locations,
                    &mut item_infos,
                    &mut properties,
                    &mut property_assocs,
                    &mut item_references,
                    &mut idat_data,
                )?;
            }
            _ => {}
        }
    }

    if brand.0 == *b"    " {
        return Err(container_err("missing ftyp box"));
    }

    Ok(HeifContainer {
        data,
        primary_item_id,
        item_locations,
        item_infos,
        properties,
        property_assocs,
        item_references,
        idat_data,
    })
}

fn parse_ftyp(
    content: &[u8],
    brand: &mut FourCC,
    compatible: &mut Vec<FourCC>,
) -> R<()> {
    if content.len() < 8 {
        return Err(container_err("ftyp too short"));
    }
    *brand = FourCC::from_bytes(&content[0..4]).unwrap();
    let mut offset = 8;
    while offset + 4 <= content.len() {
        if let Some(b) = FourCC::from_bytes(&content[offset..]) {
            compatible.push(b);
        }
        offset += 4;
    }

    let valid = [
        FourCC(*b"heic"),
        FourCC(*b"heix"),
        FourCC(*b"hevc"),
        FourCC(*b"hevx"),
        FourCC(*b"mif1"),
        FourCC(*b"msf1"),
    ];
    let is_heif =
        valid.contains(brand) || compatible.iter().any(|b| valid.contains(b));
    if !is_heif {
        return Err(container_err("not a HEIF file"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_meta<'a>(
    content: &'a [u8],
    file_data: &'a [u8],
    primary_item_id: &mut u32,
    item_locations: &mut Vec<ItemLocation>,
    item_infos: &mut Vec<ItemInfo>,
    properties: &mut Vec<ItemProperty>,
    property_assocs: &mut Vec<PropertyAssoc>,
    item_references: &mut Vec<ItemReference>,
    idat_data: &mut Option<&'a [u8]>,
) -> R<()> {
    if content.len() < 4 {
        return Err(container_err("meta box too short"));
    }
    let inner = &content[4..];

    for child in BoxIter::new(inner) {
        match child.box_type {
            FourCC::PITM => {
                *primary_item_id = parse_pitm(child.content)?;
            }
            FourCC::ILOC => {
                parse_iloc(child.content, item_locations)?;
            }
            FourCC::IINF => {
                parse_iinf(child.content, item_infos)?;
            }
            FourCC::IPRP => {
                parse_iprp(child.content, properties, property_assocs)?;
            }
            FourCC::IREF => {
                parse_iref(child.content, item_references)?;
            }
            _ => {
                // idat inside meta
                if child.box_type == FourCC::IDAT {
                    *idat_data = Some(child.content);
                }
                let _ = file_data; // suppress unused warning
            }
        }
    }
    Ok(())
}

fn parse_pitm(content: &[u8]) -> R<u32> {
    if content.len() < 4 {
        return Err(container_err("pitm too short"));
    }
    let version = content[0];
    if version == 0 {
        if content.len() < 6 {
            return Err(container_err("pitm v0 too short"));
        }
        Ok(u16::from_be_bytes([content[4], content[5]]) as u32)
    } else {
        if content.len() < 8 {
            return Err(container_err("pitm v1 too short"));
        }
        Ok(u32::from_be_bytes([content[4], content[5], content[6], content[7]]))
    }
}

fn parse_iloc(content: &[u8], locs: &mut Vec<ItemLocation>) -> R<()> {
    if content.len() < 8 {
        return Err(container_err("iloc too short"));
    }
    let version = content[0];
    let offset_size = (content[4] >> 4) & 0xF;
    let length_size = content[4] & 0xF;
    let base_offset_size = (content[5] >> 4) & 0xF;
    let index_size = if version >= 1 { content[5] & 0xF } else { 0 };

    let mut pos = 6;
    let item_count = if version < 2 {
        let c = u16::from_be_bytes([content[pos], content[pos + 1]]) as u32;
        pos += 2;
        c
    } else {
        let c = u32::from_be_bytes([
            content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
        ]);
        pos += 4;
        c
    };

    for _ in 0..item_count {
        let id_size = if version < 2 { 2usize } else { 4 };
        if pos + id_size > content.len() { break; }
        let item_id = if version < 2 {
            let id = u16::from_be_bytes([content[pos], content[pos + 1]]) as u32;
            pos += 2;
            id
        } else {
            let id = u32::from_be_bytes([
                content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
            ]);
            pos += 4;
            id
        };

        let construction_method = if version >= 1 {
            if pos + 2 > content.len() { break; }
            let m = content[pos + 1] & 0xF;
            pos += 2;
            m
        } else {
            0
        };

        if pos + 2 > content.len() { break; }
        pos += 2; // data_reference_index

        let base_offset = read_uint(content, &mut pos, base_offset_size as usize);

        if pos + 2 > content.len() { break; }
        let extent_count = u16::from_be_bytes([content[pos], content[pos + 1]]);
        pos += 2;

        let mut extents = Vec::with_capacity(extent_count as usize);
        for _ in 0..extent_count {
            if version >= 1 && index_size > 0 {
                if pos + index_size as usize > content.len() { break; }
                pos += index_size as usize;
            }
            let ext_off = read_uint(content, &mut pos, offset_size as usize);
            let ext_len = read_uint(content, &mut pos, length_size as usize);
            extents.push((ext_off, ext_len));
        }
        locs.push(ItemLocation { item_id, construction_method, base_offset, extents });
    }
    Ok(())
}

fn read_uint(data: &[u8], pos: &mut usize, size: usize) -> u64 {
    if size == 0 || *pos + size > data.len() {
        return 0;
    }
    let mut v = 0u64;
    for i in 0..size {
        v = (v << 8) | data[*pos + i] as u64;
    }
    *pos += size;
    v
}

fn parse_iinf(content: &[u8], infos: &mut Vec<ItemInfo>) -> R<()> {
    if content.len() < 6 {
        return Err(container_err("iinf too short"));
    }
    let version = content[0];
    let mut pos = 4;
    let entry_count = if version == 0 {
        let c = u16::from_be_bytes([content[pos], content[pos + 1]]) as u32;
        pos += 2;
        c
    } else {
        let c = u32::from_be_bytes([
            content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
        ]);
        pos += 4;
        c
    };
    let remaining = &content[pos..];
    let mut count = 0;
    for child in BoxIter::new(remaining) {
        if child.box_type == FourCC::INFE {
            if let Ok(info) = parse_infe(child.content) {
                infos.push(info);
                count += 1;
                if count >= entry_count { break; }
            }
        }
    }
    Ok(())
}

fn parse_infe(content: &[u8]) -> Result<ItemInfo, ()> {
    let version = *content.first().ok_or(())?;
    let min_len = match version {
        0..=1 => 8,
        2 => 12,
        _ => 14,
    };
    if content.len() < min_len { return Err(()); }
    let flags = u32::from_be_bytes([0, content[1], content[2], content[3]]);
    let hidden = (flags & 1) != 0;
    let mut pos = 4;

    let item_id = if version < 3 {
        let id = u16::from_be_bytes([content[pos], content[pos + 1]]) as u32;
        pos += 2;
        id
    } else {
        let id = u32::from_be_bytes([
            content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
        ]);
        pos += 4;
        id
    };
    pos += 2; // protection index

    let item_type = if version >= 2 {
        let ft = FourCC::from_bytes(&content[pos..]).ok_or(())?;
        pos += 4;
        ft
    } else {
        FourCC(*b"    ")
    };

    let item_name = if pos < content.len() {
        let end = content[pos..].iter().position(|&b| b == 0).unwrap_or(0);
        let name = str::from_utf8(&content[pos..pos + end]).unwrap_or("").to_string();
        pos += end + 1;
        name
    } else {
        String::new()
    };

    let content_type = if pos < content.len() {
        let end = content[pos..].iter().position(|&b| b == 0).unwrap_or(0);
        str::from_utf8(&content[pos..pos + end]).unwrap_or("").to_string()
    } else {
        String::new()
    };

    Ok(ItemInfo { item_id, item_type, item_name, content_type, hidden })
}

fn parse_iprp(
    content: &[u8],
    props: &mut Vec<ItemProperty>,
    assocs: &mut Vec<PropertyAssoc>,
) -> R<()> {
    for child in BoxIter::new(content) {
        match child.box_type {
            FourCC::IPCO => parse_ipco(child.content, props)?,
            FourCC::IPMA => parse_ipma(child.content, assocs)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_ipco(content: &[u8], props: &mut Vec<ItemProperty>) -> R<()> {
    for child in BoxIter::new(content) {
        let prop = match child.box_type {
            FourCC::ISPE => parse_ispe(child.content)
                .map(|(w, h)| ItemProperty::ImageExtents { width: w, height: h })
                .unwrap_or(ItemProperty::Unknown),
            FourCC::HVCC | FourCC::HVCB => parse_hvcc(child.content)
                .map(ItemProperty::HevcConfig)
                .unwrap_or(ItemProperty::Unknown),
            FourCC::COLR => parse_colr(child.content)
                .map(ItemProperty::ColorInfo)
                .unwrap_or(ItemProperty::Unknown),
            FourCC::CLAP => parse_clap(child.content)
                .map(ItemProperty::CleanAperture)
                .unwrap_or(ItemProperty::Unknown),
            FourCC::IROT => parse_irot(child.content)
                .map(ItemProperty::Rotation)
                .unwrap_or(ItemProperty::Unknown),
            FourCC::IMIR => parse_imir(child.content)
                .map(ItemProperty::Mirror)
                .unwrap_or(ItemProperty::Unknown),
            FourCC::AUXC => parse_auxc(child.content)
                .map(ItemProperty::AuxiliaryType)
                .unwrap_or(ItemProperty::Unknown),
            _ => ItemProperty::Unknown,
        };
        props.push(prop);
    }
    Ok(())
}

fn parse_ispe(c: &[u8]) -> Option<(u32, u32)> {
    if c.len() < 12 { return None; }
    let w = u32::from_be_bytes([c[4], c[5], c[6], c[7]]);
    let h = u32::from_be_bytes([c[8], c[9], c[10], c[11]]);
    Some((w, h))
}

fn parse_hvcc(c: &[u8]) -> Option<HvcCConfig> {
    if c.len() < 23 { return None; }
    let chroma_format = c[16] & 0x3;
    let bit_depth_luma_minus8 = c[17] & 0x7;
    let length_size_minus_one = c[21] & 0x3;
    let num_arrays = c[22];
    let mut pos = 23;
    let mut nal_units = Vec::new();
    for _ in 0..num_arrays {
        if pos + 3 > c.len() { break; }
        pos += 1; // nal_unit_type / array_completeness
        let num_nalus = u16::from_be_bytes([c[pos], c[pos + 1]]);
        pos += 2;
        for _ in 0..num_nalus {
            if pos + 2 > c.len() { break; }
            let nalu_len = u16::from_be_bytes([c[pos], c[pos + 1]]) as usize;
            pos += 2;
            if pos + nalu_len > c.len() { break; }
            nal_units.push(c[pos..pos + nalu_len].to_vec());
            pos += nalu_len;
        }
    }
    Some(HvcCConfig { chroma_format, bit_depth_luma_minus8, length_size_minus_one, nal_units })
}

fn parse_colr(c: &[u8]) -> Option<ColorInfoKind> {
    if c.len() < 4 { return None; }
    match &c[0..4] {
        b"nclx" => {
            if c.len() < 11 { return None; }
            Some(ColorInfoKind::Nclx {
                color_primaries: u16::from_be_bytes([c[4], c[5]]),
                transfer_characteristics: u16::from_be_bytes([c[6], c[7]]),
                matrix_coefficients: u16::from_be_bytes([c[8], c[9]]),
                full_range: (c[10] >> 7) != 0,
            })
        }
        b"prof" | b"ricc" => Some(ColorInfoKind::IccProfile(c[4..].to_vec())),
        _ => None,
    }
}

fn parse_clap(c: &[u8]) -> Option<CleanAperture> {
    if c.len() < 32 { return None; }
    Some(CleanAperture {
        width_n: u32::from_be_bytes([c[0], c[1], c[2], c[3]]),
        width_d: u32::from_be_bytes([c[4], c[5], c[6], c[7]]),
        height_n: u32::from_be_bytes([c[8], c[9], c[10], c[11]]),
        height_d: u32::from_be_bytes([c[12], c[13], c[14], c[15]]),
        horiz_off_n: i32::from_be_bytes([c[16], c[17], c[18], c[19]]),
        horiz_off_d: u32::from_be_bytes([c[20], c[21], c[22], c[23]]),
        vert_off_n: i32::from_be_bytes([c[24], c[25], c[26], c[27]]),
        vert_off_d: u32::from_be_bytes([c[28], c[29], c[30], c[31]]),
    })
}

fn parse_irot(c: &[u8]) -> Option<ImageRotation> {
    let angle = match c.first()? & 0x03 {
        0 => 0,
        1 => 270,
        2 => 180,
        3 => 90,
        _ => 0,
    };
    Some(ImageRotation { angle })
}

fn parse_imir(c: &[u8]) -> Option<ImageMirror> {
    Some(ImageMirror { axis: c.first()? & 0x01 })
}

fn parse_auxc(c: &[u8]) -> Option<String> {
    if c.len() < 5 { return None; }
    let data = &c[4..];
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    Some(str::from_utf8(&data[..end]).unwrap_or("").to_string())
}

fn parse_ipma(content: &[u8], assocs: &mut Vec<PropertyAssoc>) -> R<()> {
    if content.len() < 8 { return Err(container_err("ipma too short")); }
    let version = content[0];
    let flags = u32::from_be_bytes([0, content[1], content[2], content[3]]);
    let mut pos = 4;
    let entry_count = u32::from_be_bytes([
        content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
    ]);
    pos += 4;

    for _ in 0..entry_count {
        let id_size = if version < 1 { 2 } else { 4 };
        if pos + id_size > content.len() { break; }
        let item_id = if version < 1 {
            let id = u16::from_be_bytes([content[pos], content[pos + 1]]) as u32;
            pos += 2;
            id
        } else {
            let id = u32::from_be_bytes([
                content[pos], content[pos + 1], content[pos + 2], content[pos + 3],
            ]);
            pos += 4;
            id
        };
        if pos >= content.len() { break; }
        let assoc_count = content[pos];
        pos += 1;
        let mut properties = Vec::with_capacity(assoc_count as usize);
        for _ in 0..assoc_count {
            if pos >= content.len() { break; }
            let (essential, prop_idx) = if (flags & 1) != 0 {
                if pos + 2 > content.len() { break; }
                let v = u16::from_be_bytes([content[pos], content[pos + 1]]);
                pos += 2;
                ((v >> 15) != 0, v & 0x7FFF)
            } else {
                let v = content[pos];
                pos += 1;
                ((v >> 7) != 0, (v & 0x7F) as u16)
            };
            properties.push((prop_idx, essential));
        }
        assocs.push(PropertyAssoc { item_id, properties });
    }
    Ok(())
}

fn parse_iref(content: &[u8], refs: &mut Vec<ItemReference>) -> R<()> {
    if content.len() < 4 { return Err(container_err("iref too short")); }
    let version = content[0];
    let remaining = &content[4..];
    for child in BoxIter::new(remaining) {
        let ref_type = child.box_type;
        let data = child.content;
        let mut pos = 0;
        while pos < data.len() {
            let id_size = if version == 0 { 2usize } else { 4 };
            if pos + id_size > data.len() { break; }
            let from_id = if version == 0 {
                let id = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
                pos += 2;
                id
            } else {
                let id = u32::from_be_bytes([
                    data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                ]);
                pos += 4;
                id
            };
            if pos + 2 > data.len() { break; }
            let ref_count = u16::from_be_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            let mut to_ids = Vec::with_capacity(ref_count as usize);
            for _ in 0..ref_count {
                if pos + id_size > data.len() { break; }
                let to_id = if version == 0 {
                    let id = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
                    pos += 2;
                    id
                } else {
                    let id = u32::from_be_bytes([
                        data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                    ]);
                    pos += 4;
                    id
                };
                to_ids.push(to_id);
            }
            refs.push(ItemReference { reference_type: ref_type, from_item_id: from_id, to_item_ids: to_ids });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// HEVC helpers
// ---------------------------------------------------------------------------

fn hevc_decode_item_data(item: &Item, data: &[u8]) -> R<DecodedFrame> {
    if let Some(ref cfg) = item.hevc_config {
        hevc::decode_with_config(&cfg.to_bhd(), data)
            .map_err(DecodeError::HevcDecode)
    } else {
        hevc::decode(data).map_err(DecodeError::HevcDecode)
    }
}

// ---------------------------------------------------------------------------
// Decode pipeline
// ---------------------------------------------------------------------------

fn apply_color_info(frame: &mut DecodedFrame, color_info: &Option<ColorInfoKind>) {
    if let Some(ColorInfoKind::Nclx { full_range, matrix_coefficients, .. }) = color_info {
        frame.full_range = *full_range;
        frame.matrix_coeffs = *matrix_coefficients as u8;
    }
}

fn apply_transforms(frame: DecodedFrame, transforms: &[Transform]) -> DecodedFrame {
    let mut f = frame;
    for t in transforms {
        f = match t {
            Transform::CleanAperture(clap) => {
                apply_clean_aperture(f, clap)
            }
            Transform::Mirror(m) => match m.axis {
                0 => f.mirror_vertical(),
                1 => f.mirror_horizontal(),
                _ => f,
            },
            Transform::Rotation(r) => match r.angle {
                90 => f.rotate_90_cw(),
                180 => f.rotate_180(),
                270 => f.rotate_270_cw(),
                _ => f,
            },
        };
    }
    f
}

fn decode_item(
    container: &HeifContainer<'_>,
    item: &Item,
    depth: u32,
) -> R<DecodedFrame> {
    if depth > 8 {
        return Err(invalid_data("derived image reference chain too deep"));
    }
    let mut frame = match item.kind {
        ItemKind::Grid => decode_grid(container, item)?,
        ItemKind::Iden => decode_iden(container, item, depth)?,
        ItemKind::Iovl => decode_iovl(container, item, depth)?,
        _ => {
            let data = container
                .get_item_data(item.id)
                .ok_or_else(|| invalid_data("missing image data"))?;
            hevc_decode_item_data(item, data)?
        }
    };
    apply_color_info(&mut frame, &item.color_info);
    let frame = apply_transforms(frame, &item.transforms);
    Ok(frame)
}

fn decode_iden(
    container: &HeifContainer<'_>,
    item: &Item,
    depth: u32,
) -> R<DecodedFrame> {
    let source_ids = container.get_item_references(item.id, FourCC::DIMG);
    let &source_id = source_ids.first().ok_or_else(|| invalid_data("iden has no dimg reference"))?;
    let source = container.get_item(source_id).ok_or_else(|| invalid_data("iden dimg target not found"))?;
    decode_item(container, &source, depth + 1)
}

fn decode_grid(container: &HeifContainer<'_>, grid_item: &Item) -> R<DecodedFrame> {
    let grid_data = container
        .get_item_data(grid_item.id)
        .ok_or_else(|| invalid_data("missing grid descriptor"))?;
    if grid_data.len() < 8 {
        return Err(invalid_data("grid descriptor too short"));
    }
    let flags = grid_data[1];
    let rows = grid_data[2] as u32 + 1;
    let cols = grid_data[3] as u32 + 1;
    let (output_width, output_height) = if (flags & 1) != 0 {
        if grid_data.len() < 12 {
            return Err(invalid_data("grid descriptor too short for 32-bit dims"));
        }
        (
            u32::from_be_bytes([grid_data[4], grid_data[5], grid_data[6], grid_data[7]]),
            u32::from_be_bytes([grid_data[8], grid_data[9], grid_data[10], grid_data[11]]),
        )
    } else {
        (
            u16::from_be_bytes([grid_data[4], grid_data[5]]) as u32,
            u16::from_be_bytes([grid_data[6], grid_data[7]]) as u32,
        )
    };

    let tile_ids = container.get_item_references(grid_item.id, FourCC::DIMG);
    let expected = (rows * cols) as usize;
    if tile_ids.len() != expected {
        return Err(invalid_data("grid tile count mismatch"));
    }

    let first_tile = container.get_item(tile_ids[0]).ok_or_else(|| invalid_data("missing tile item"))?;
    let tile_cfg = first_tile.hevc_config.as_ref().ok_or_else(|| invalid_data("missing tile hvcC config"))?;
    let (tile_width, tile_height) = first_tile.dimensions.ok_or_else(|| invalid_data("missing tile dimensions"))?;

    let bit_depth = tile_cfg.bit_depth_luma_minus8 + 8;
    let chroma_format = tile_cfg.chroma_format;
    let mut output = DecodedFrame::with_params(output_width, output_height, bit_depth, chroma_format);

    let tile_data_list: Vec<&[u8]> = tile_ids
        .iter()
        .map(|&tid| {
            container.get_item_data(tid).ok_or_else(|| invalid_data("missing tile data"))
        })
        .collect::<R<_>>()?;

    let decoded_tiles: Vec<DecodedFrame> = tile_data_list
        .iter()
        .map(|tile_data| {
            hevc::decode_with_config(&tile_cfg.to_bhd(), tile_data)
                .map_err(DecodeError::HevcDecode)
        })
        .collect::<R<_>>()?;

    for (tile_idx, tile_frame) in decoded_tiles.iter().enumerate() {
        if tile_idx == 0 {
            output.full_range = tile_frame.full_range;
            output.matrix_coeffs = tile_frame.matrix_coeffs;
        }
        let tile_row = tile_idx as u32 / cols;
        let tile_col = tile_idx as u32 % cols;
        let dst_x = tile_col * tile_width;
        let dst_y = tile_row * tile_height;

        let copy_w = tile_frame.cropped_width().min(output_width.saturating_sub(dst_x));
        let copy_h = tile_frame.cropped_height().min(output_height.saturating_sub(dst_y));
        let src_x_start = tile_frame.crop_left;
        let src_y_start = tile_frame.crop_top;

        // Luma
        for row in 0..copy_h {
            let src_row = (src_y_start + row) as usize;
            let dst_row = (dst_y + row) as usize;
            for col in 0..copy_w {
                let src_col = (src_x_start + col) as usize;
                let dst_col = (dst_x + col) as usize;
                let y_stride = output.y_stride();
                output.y_plane[dst_row * y_stride + dst_col] =
                    tile_frame.y_plane[src_row * tile_frame.y_stride() + src_col];
            }
        }
        // Chroma
        if chroma_format > 0 {
            let (sub_x, sub_y) = match chroma_format {
                1 => (2u32, 2u32),
                2 => (2, 1),
                3 => (1, 1),
                _ => (2, 2),
            };
            let c_copy_w = copy_w.div_ceil(sub_x);
            let c_copy_h = copy_h.div_ceil(sub_y);
            let c_dst_x = dst_x / sub_x;
            let c_dst_y = dst_y / sub_y;
            let c_src_x = src_x_start / sub_x;
            let c_src_y = src_y_start / sub_y;
            let src_cs = tile_frame.c_stride();
            let dst_cs = output.c_stride();
            for row in 0..c_copy_h {
                let sr = (c_src_y + row) as usize;
                let dr = (c_dst_y + row) as usize;
                for col in 0..c_copy_w {
                    let sc = (c_src_x + col) as usize;
                    let dc = (c_dst_x + col) as usize;
                    let si = sr * src_cs + sc;
                    let di = dr * dst_cs + dc;
                    if si < tile_frame.cb_plane.len() && di < output.cb_plane.len() {
                        output.cb_plane[di] = tile_frame.cb_plane[si];
                        output.cr_plane[di] = tile_frame.cr_plane[si];
                    }
                }
            }
        }
    }
    Ok(output)
}

fn decode_iovl(
    container: &HeifContainer<'_>,
    iovl_item: &Item,
    depth: u32,
) -> R<DecodedFrame> {
    let iovl_data = container
        .get_item_data(iovl_item.id)
        .ok_or_else(|| invalid_data("missing overlay descriptor"))?;
    if iovl_data.len() < 6 {
        return Err(invalid_data("overlay descriptor too short"));
    }
    let flags = iovl_data[1];
    let large = (flags & 1) != 0;

    let tile_ids = container.get_item_references(iovl_item.id, FourCC::DIMG);
    if tile_ids.is_empty() {
        return Err(invalid_data("overlay has no tile references"));
    }

    let off_size = if large { 4usize } else { 2 };
    let per_tile = 2 * off_size;
    let fixed_end = 4 + 2 * off_size;
    let tile_data_size = tile_ids.len() * per_tile;
    let fill_bytes = iovl_data
        .len()
        .checked_sub(fixed_end + tile_data_size)
        .ok_or_else(|| invalid_data("overlay descriptor too short for tiles"))?;

    let num_fill_channels = fill_bytes / 2;
    let mut fill_values = [0u16; 4];
    for i in 0..num_fill_channels.min(4) {
        fill_values[i] = u16::from_be_bytes([iovl_data[4 + i * 2], iovl_data[4 + i * 2 + 1]]);
    }
    let mut pos = 4 + fill_bytes;

    let (canvas_width, canvas_height) = if large {
        if pos + 8 > iovl_data.len() {
            return Err(invalid_data("overlay descriptor truncated"));
        }
        let w = u32::from_be_bytes([
            iovl_data[pos], iovl_data[pos + 1], iovl_data[pos + 2], iovl_data[pos + 3],
        ]);
        let h = u32::from_be_bytes([
            iovl_data[pos + 4], iovl_data[pos + 5], iovl_data[pos + 6], iovl_data[pos + 7],
        ]);
        pos += 8;
        (w, h)
    } else {
        if pos + 4 > iovl_data.len() {
            return Err(invalid_data("overlay descriptor truncated"));
        }
        let w = u16::from_be_bytes([iovl_data[pos], iovl_data[pos + 1]]) as u32;
        let h = u16::from_be_bytes([iovl_data[pos + 2], iovl_data[pos + 3]]) as u32;
        pos += 4;
        (w, h)
    };

    let mut offsets = Vec::with_capacity(tile_ids.len());
    for _ in 0..tile_ids.len() {
        let (x, y) = if large {
            if pos + 8 > iovl_data.len() {
                return Err(invalid_data("overlay offset data truncated"));
            }
            let x = i32::from_be_bytes([
                iovl_data[pos], iovl_data[pos + 1], iovl_data[pos + 2], iovl_data[pos + 3],
            ]);
            let y = i32::from_be_bytes([
                iovl_data[pos + 4], iovl_data[pos + 5], iovl_data[pos + 6], iovl_data[pos + 7],
            ]);
            pos += 8;
            (x, y)
        } else {
            if pos + 4 > iovl_data.len() {
                return Err(invalid_data("overlay offset data truncated"));
            }
            let x = i16::from_be_bytes([iovl_data[pos], iovl_data[pos + 1]]) as i32;
            let y = i16::from_be_bytes([iovl_data[pos + 2], iovl_data[pos + 3]]) as i32;
            pos += 4;
            (x, y)
        };
        offsets.push((x, y));
    }

    let first_tile_item = container.get_item(tile_ids[0]).ok_or_else(|| invalid_data("missing overlay tile item"))?;
    let first_tile_cfg = first_tile_item.hevc_config.as_ref().ok_or_else(|| invalid_data("missing overlay tile hvcC"))?;
    let bit_depth = first_tile_cfg.bit_depth_luma_minus8 + 8;
    let chroma_format = first_tile_cfg.chroma_format;

    let mut output = DecodedFrame::with_params(canvas_width, canvas_height, bit_depth, chroma_format);
    let fill_shift = 16u32.saturating_sub(bit_depth as u32);
    let y_fill = fill_values[0] >> fill_shift;
    let cb_fill = if num_fill_channels > 1 { fill_values[1] >> fill_shift } else { 1u16 << (bit_depth - 1) };
    let cr_fill = if num_fill_channels > 2 { fill_values[2] >> fill_shift } else { 1u16 << (bit_depth - 1) };
    output.y_plane.fill(y_fill);
    output.cb_plane.fill(cb_fill);
    output.cr_plane.fill(cr_fill);

    for (idx, &tile_id) in tile_ids.iter().enumerate() {
        let tile_item = container.get_item(tile_id).ok_or_else(|| invalid_data("missing overlay tile"))?;
        let tile_frame = decode_item(container, &tile_item, depth + 1)?;
        if idx == 0 {
            output.full_range = tile_frame.full_range;
            output.matrix_coeffs = tile_frame.matrix_coeffs;
        }
        let (off_x, off_y) = offsets[idx];
        let dst_x = off_x.max(0) as u32;
        let dst_y = off_y.max(0) as u32;
        let copy_w = tile_frame.cropped_width().min(canvas_width.saturating_sub(dst_x));
        let copy_h = tile_frame.cropped_height().min(canvas_height.saturating_sub(dst_y));

        for row in 0..copy_h {
            let src_row = (tile_frame.crop_top + row) as usize;
            let dst_row = (dst_y + row) as usize;
            for col in 0..copy_w {
                let src_col = (tile_frame.crop_left + col) as usize;
                let dst_col = (dst_x + col) as usize;
                let si = src_row * tile_frame.y_stride() + src_col;
                let di = dst_row * output.y_stride() + dst_col;
                if si < tile_frame.y_plane.len() && di < output.y_plane.len() {
                    output.y_plane[di] = tile_frame.y_plane[si];
                }
            }
        }
        if chroma_format > 0 {
            let (sub_x, sub_y) = match chroma_format {
                1 => (2u32, 2u32),
                2 => (2, 1),
                3 => (1, 1),
                _ => (2, 2),
            };
            let c_copy_w = copy_w.div_ceil(sub_x);
            let c_copy_h = copy_h.div_ceil(sub_y);
            let c_dst_x = dst_x / sub_x;
            let c_dst_y = dst_y / sub_y;
            let c_src_x = tile_frame.crop_left / sub_x;
            let c_src_y = tile_frame.crop_top / sub_y;
            let src_cs = tile_frame.c_stride();
            let dst_cs = output.c_stride();
            for row in 0..c_copy_h {
                let sr = (c_src_y + row) as usize;
                let dr = (c_dst_y + row) as usize;
                for col in 0..c_copy_w {
                    let sc = (c_src_x + col) as usize;
                    let dc = (c_dst_x + col) as usize;
                    let si = sr * src_cs + sc;
                    let di = dr * dst_cs + dc;
                    if si < tile_frame.cb_plane.len() && di < output.cb_plane.len() {
                        output.cb_plane[di] = tile_frame.cb_plane[si];
                        output.cr_plane[di] = tile_frame.cr_plane[si];
                    }
                }
            }
        }
    }
    Ok(output)
}

fn decode_alpha_plane(
    container: &HeifContainer<'_>,
    alpha_id: u32,
    primary: &DecodedFrame,
) -> Option<Vec<u16>> {
    let alpha_item = container.get_item(alpha_id)?;
    let alpha_data = container.get_item_data(alpha_id)?;
    let alpha_frame = hevc_decode_item_data(&alpha_item, alpha_data).ok()?;

    let pw = primary.cropped_width();
    let ph = primary.cropped_height();
    let aw = alpha_frame.cropped_width();
    let ah = alpha_frame.cropped_height();
    let total = (pw * ph) as usize;
    let mut out = Vec::with_capacity(total);

    if aw == pw && ah == ph {
        let ys = alpha_frame.crop_top;
        let xs = alpha_frame.crop_left;
        for y in 0..ph {
            for x in 0..pw {
                let idx = ((ys + y) * alpha_frame.width + (xs + x)) as usize;
                out.push(alpha_frame.y_plane[idx]);
            }
        }
    } else {
        for dy in 0..ph {
            for dx in 0..pw {
                let sx = (dx as f64) * (aw as f64 - 1.0) / (pw as f64 - 1.0).max(1.0);
                let sy = (dy as f64) * (ah as f64 - 1.0) / (ph as f64 - 1.0).max(1.0);
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let x1 = (x0 + 1).min(aw - 1);
                let y1 = (y0 + 1).min(ah - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;
                let s = alpha_frame.width;
                let ox = alpha_frame.crop_left;
                let oy = alpha_frame.crop_top;
                let get = |px: u32, py: u32| -> f64 {
                    let idx = ((oy + py) * s + (ox + px)) as usize;
                    alpha_frame.y_plane.get(idx).copied().unwrap_or(0) as f64
                };
                let v = get(x0, y0) * (1.0 - fx) * (1.0 - fy)
                    + get(x1, y0) * fx * (1.0 - fy)
                    + get(x0, y1) * (1.0 - fx) * fy
                    + get(x1, y1) * fx * fy;
                out.push(v.round() as u16);
            }
        }
    }
    Some(out)
}

fn apply_clean_aperture(frame: DecodedFrame, clap: &CleanAperture) -> DecodedFrame {
    let mut f = frame;
    let conf_w = f.cropped_width();
    let conf_h = f.cropped_height();
    let clean_w = if clap.width_d > 0 { clap.width_n / clap.width_d } else { conf_w };
    let clean_h = if clap.height_d > 0 { clap.height_n / clap.height_d } else { conf_h };
    if clean_w >= conf_w && clean_h >= conf_h {
        return f;
    }
    let horiz = if clap.horiz_off_d > 0 { (clap.horiz_off_n as f64) / (clap.horiz_off_d as f64) } else { 0.0 };
    let vert = if clap.vert_off_d > 0 { (clap.vert_off_n as f64) / (clap.vert_off_d as f64) } else { 0.0 };
    let extra_left = ((conf_w as f64 - clean_w as f64) / 2.0 + horiz).round() as u32;
    let extra_top = ((conf_h as f64 - clean_h as f64) / 2.0 + vert).round() as u32;
    let extra_right = conf_w.saturating_sub(clean_w).saturating_sub(extra_left);
    let extra_bottom = conf_h.saturating_sub(clean_h).saturating_sub(extra_top);
    f.crop_left += extra_left;
    f.crop_right += extra_right;
    f.crop_top += extra_top;
    f.crop_bottom += extra_bottom;
    f
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Parse HEIC/HEIF metadata without decoding pixel data.
pub fn get_heic_image_info(data: &[u8]) -> Result<HeicImageInfo, DecodeError> {
    let container = parse_container(data)?;
    let primary = container.primary_item().ok_or_else(|| invalid_data("no primary image"))?;

    let has_alpha = !container.find_auxiliary_items(primary.id, "urn:mpeg:hevc:2015:auxid:1").is_empty()
        || !container.find_auxiliary_items(primary.id, "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha").is_empty();

    let has_exif = container.item_infos.iter().any(|i| i.item_type == FourCC(*b"Exif"));
    let has_xmp = container.item_infos.iter().any(|i| {
        i.item_type == FourCC(*b"mime")
            && (i.content_type.contains("xmp") || i.content_type.contains("rdf+xml"))
    });
    let has_thumbnail = !container.find_thumbnails(primary.id).is_empty();

    if let Some(ref cfg) = primary.hevc_config {
        let bhd_cfg = cfg.to_bhd();
        if let Ok(info) = hevc::get_info_from_config(&bhd_cfg) {
            return Ok(HeicImageInfo {
                width: info.width,
                height: info.height,
                has_alpha,
                bit_depth: cfg.bit_depth_luma_minus8 + 8,
                chroma_format: cfg.chroma_format,
                has_exif,
                has_xmp,
                has_thumbnail,
            });
        }
    }

    // Grid/iden/iovl: dimensions from ispe
    if primary.kind != ItemKind::Hvc1 {
        if let Some((w, h)) = primary.dimensions {
            let (bit_depth, chroma_format) = tile_format_from_dimg(&container, primary.id);
            return Ok(HeicImageInfo {
                width: w,
                height: h,
                has_alpha,
                bit_depth,
                chroma_format,
                has_exif,
                has_xmp,
                has_thumbnail,
            });
        }
    }

    // Fallback: parse raw HEVC stream
    let image_data = container.get_item_data(primary.id)
        .ok_or_else(|| invalid_data("missing image data"))?;
    let hevc_info = hevc::get_info(image_data).map_err(DecodeError::HevcDecode)?;
    Ok(HeicImageInfo {
        width: hevc_info.width,
        height: hevc_info.height,
        has_alpha,
        bit_depth: 8,
        chroma_format: 1,
        has_exif,
        has_xmp,
        has_thumbnail,
    })
}

fn tile_format_from_dimg(container: &HeifContainer<'_>, from_id: u32) -> (u8, u8) {
    for r in &container.item_references {
        if r.reference_type == FourCC::DIMG && r.from_item_id == from_id {
            if let Some(&tile_id) = r.to_item_ids.first() {
                if let Some(tile) = container.get_item(tile_id) {
                    if let Some(ref cfg) = tile.hevc_config {
                        return (cfg.bit_depth_luma_minus8 + 8, cfg.chroma_format);
                    }
                }
            }
        }
    }
    (8, 1) // fallback
}

/// Decode a HEIC/HEIF file to a raw planar YCbCr frame.
pub fn decode_heic_to_frame(data: &[u8]) -> Result<DecodedFrame, DecodeError> {
    let container = parse_container(data)?;
    let primary = container.primary_item().ok_or_else(|| invalid_data("no primary image"))?;
    let mut frame = decode_item(&container, &primary, 0)?;

    // Decode alpha
    let alpha_id = container
        .find_auxiliary_items(primary.id, "urn:mpeg:hevc:2015:auxid:1")
        .first()
        .copied()
        .or_else(|| {
            container
                .find_auxiliary_items(primary.id, "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")
                .first()
                .copied()
        });
    if let Some(aid) = alpha_id {
        if let Some(alpha) = decode_alpha_plane(&container, aid, &frame) {
            frame.alpha_plane = Some(alpha);
        }
    }
    Ok(frame)
}

/// Decode a HEIC/HEIF file to interleaved pixel data.
pub fn decode_heic(data: &[u8], layout: PixelLayout) -> Result<DecodeOutput, DecodeError> {
    let frame = decode_heic_to_frame(data)?;
    let width = frame.cropped_width();
    let height = frame.cropped_height();
    let pixels = match layout {
        PixelLayout::Rgb8 => frame.to_rgb(),
        PixelLayout::Rgba8 => frame.to_rgba(),
        PixelLayout::Bgr8 => frame.to_bgr(),
        PixelLayout::Bgra8 => frame.to_bgra(),
    };
    Ok(DecodeOutput { data: pixels, width, height, layout })
}

/// Decode the embedded thumbnail from a HEIC/HEIF file, if present.
///
/// Returns `None` when the file contains no thumbnail.
pub fn decode_heic_thumbnail(data: &[u8], layout: PixelLayout) -> Result<Option<DecodeOutput>, DecodeError> {
    let container = parse_container(data)?;
    let primary = container.primary_item().ok_or_else(|| invalid_data("no primary image"))?;

    let thumb_ids = container.find_thumbnails(primary.id);
    let Some(&thumb_id) = thumb_ids.first() else {
        return Ok(None);
    };
    let thumb_item = container.get_item(thumb_id).ok_or_else(|| invalid_data("thumbnail item not found"))?;
    let frame = decode_item(&container, &thumb_item, 0)?;
    let width = frame.cropped_width();
    let height = frame.cropped_height();
    let pixels = match layout {
        PixelLayout::Rgb8 => frame.to_rgb(),
        PixelLayout::Rgba8 => frame.to_rgba(),
        PixelLayout::Bgr8 => frame.to_bgr(),
        PixelLayout::Bgra8 => frame.to_bgra(),
    };
    Ok(Some(DecodeOutput { data: pixels, width, height, layout }))
}
