//! Intra-prediction primitive sub-table.

pub mod angs;

/// Function pointer type for batched all-angular intra prediction.
pub type PredAllAngsFn = fn(&mut [u16], &[i32], &[i32], usize, u8, u8, u8);

/// Intra-prediction primitives. Currently covers the rough-search angular batch;
/// exact planar/DC/angular prediction for committed recon stays in the decoder.
/// Function pointer type for batched all-angular intra prediction to u8 (8-bit only).
pub type PredAllAngsU8Fn = fn(&mut [u8], &[i32], &[i32], usize, u8, u8, u8);

pub struct IntraPrimitives {
    /// Batched angular intra prediction (modes 2..=34) for the rough search.
    /// `(dst, unfiltered_border, filtered_border, center, log2_size, c_idx, bit_depth)`.
    pub pred_allangs: PredAllAngsFn,
    /// Same batch prediction but narrows directly to u8 output. Only valid for `bit_depth == 8`.
    pub pred_allangs_u8: PredAllAngsU8Fn,
}
