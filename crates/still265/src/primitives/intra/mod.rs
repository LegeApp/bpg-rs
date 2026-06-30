//! Intra-prediction primitive sub-table.

pub mod angs;
pub mod exact;

/// Function pointer type for batched all-angular intra prediction.
pub type PredAllAngsFn = fn(&mut [u16], &[i32], &[i32], usize, u8, u8, u8);

/// Function pointer type for batched all-angular intra prediction to u8 (8-bit only).
pub type PredAllAngsU8Fn = fn(&mut [u8], &[i32], &[i32], usize, u8, u8, u8);

/// 10/12-bit-capable exact planar/DC intra prediction from one prepared border.
pub type PredExactU16Fn = fn(&mut [u16], &[i32], usize, u8, u8, u8);
/// 10/12-bit-capable exact angular prediction from one prepared border and a single mode.
pub type PredAngularU16Fn = fn(&mut [u16], &[i32], usize, u8, u8, u8, u8);
/// 8-bit exact planar/DC intra prediction from one prepared border.
pub type PredExactU8Fn = fn(&mut [u8], &[i32], usize, u8, u8, u8);
/// 8-bit exact angular intra prediction from one prepared border and a single mode.
pub type PredAngularU8Fn = fn(&mut [u8], &[i32], usize, u8, u8, u8, u8);

/// Intra-prediction primitives. Rough search uses the all-angular batch; exact
/// RDO uses the single-mode planar/DC/angular slots.
pub struct IntraPrimitives {
    /// Batched angular intra prediction (modes 2..=34) for the rough search.
    /// `(dst, unfiltered_border, filtered_border, center, log2_size, c_idx, bit_depth)`.
    pub pred_allangs: PredAllAngsFn,
    /// Same batch prediction but narrows directly to u8 output. Only valid for `bit_depth == 8`.
    pub pred_allangs_u8: PredAllAngsU8Fn,
    /// Exact planar prediction to u16.
    #[allow(dead_code)]
    pub pred_planar_u16: PredExactU16Fn,
    /// Exact DC prediction to u16.
    #[allow(dead_code)]
    pub pred_dc_u16: PredExactU16Fn,
    /// Exact angular prediction to u16.
    #[allow(dead_code)]
    pub pred_angular_u16: PredAngularU16Fn,
    /// Exact planar prediction, narrowed directly to u8.
    pub pred_planar_u8: PredExactU8Fn,
    /// Exact DC prediction, narrowed directly to u8.
    pub pred_dc_u8: PredExactU8Fn,
    /// Exact angular prediction, narrowed directly to u8.
    pub pred_angular_u8: PredAngularU8Fn,
}
