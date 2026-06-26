//! Geometry descriptors passed through StillSearch.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CuGeom {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) log2_size: u8,
    pub(super) depth: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TuGeom {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) log2_size: u8,
    pub(super) trafo_depth: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PuGeom {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) log2_size: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ChromaGeom {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) log2_size: u8,
    pub(super) count: u8,
}
