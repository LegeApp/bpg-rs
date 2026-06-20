//! Explicit residual pricing boundary for rdo2 block evaluations.

use crate::contexts::Contexts;
use crate::residual::{estimate_residual_bits_into, get_scan_order, ResidualPricingScratch};

use super::super::Encoder;
use super::policy::{EvalPolicy, ResidualBitPolicy};

pub(in crate::encoder) struct ResidualPricer;

impl ResidualPricer {
    pub(in crate::encoder) fn price(
        enc: &mut Encoder<'_>,
        ctxs: &Contexts,
        levels: &[i16],
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        policy: EvalPolicy,
        scratch: &mut ResidualPricingScratch,
    ) -> u64 {
        if levels.iter().all(|&level| level == 0) {
            return 0;
        }

        match policy.bits {
            ResidualBitPolicy::Approx => {
                enc.stats.rdo2_residual_approx_pricings += 1;
                enc.trace.note_rdo2_residual_pricing(false);
                enc.approx_residual_frac_bits(ctxs, levels, log2_size, c_idx)
            }
            ResidualBitPolicy::Exact => {
                enc.stats.rdo2_residual_exact_pricings += 1;
                enc.trace.note_rdo2_residual_pricing(true);
                enc.stats.residual_bit_estimates += 1;
                let t = enc.prof.on.then(std::time::Instant::now);
                let scan = get_scan_order(log2_size, mode, c_idx, enc.cat);
                let bits = estimate_residual_bits_into(
                    ctxs,
                    levels,
                    log2_size,
                    c_idx,
                    scan,
                    enc.sign_data_hiding,
                    scratch,
                );
                if let Some(t) = t {
                    enc.prof.residual_bits += t.elapsed();
                }
                bits
            }
        }
    }
}
