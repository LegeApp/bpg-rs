//! Typed CTU-local arenas for plans, coefficients, and recon overlays.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlanId(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CoeffId(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReconId(pub(super) u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ArenaMark {
    plans: usize,
    coeffs: usize,
    recons: usize,
}

#[derive(Default, Debug)]
pub(super) struct CtuArenas<P = (), C = (), R = ()> {
    plans: Vec<P>,
    coeffs: Vec<C>,
    recons: Vec<R>,
}

impl<P, C, R> CtuArenas<P, C, R> {
    pub(super) fn clear(&mut self) {
        self.plans.clear();
        self.coeffs.clear();
        self.recons.clear();
    }

    pub(super) fn mark(&self) -> ArenaMark {
        ArenaMark {
            plans: self.plans.len(),
            coeffs: self.coeffs.len(),
            recons: self.recons.len(),
        }
    }

    pub(super) fn rewind(&mut self, mark: ArenaMark) {
        self.plans.truncate(mark.plans);
        self.coeffs.truncate(mark.coeffs);
        self.recons.truncate(mark.recons);
    }

    pub(super) fn push_plan(&mut self, plan: P) -> PlanId {
        let id = PlanId(u32::try_from(self.plans.len()).expect("CTU plan arena overflow"));
        self.plans.push(plan);
        id
    }

    pub(super) fn push_coeffs(&mut self, coeffs: C) -> CoeffId {
        let id = CoeffId(u32::try_from(self.coeffs.len()).expect("CTU coeff arena overflow"));
        self.coeffs.push(coeffs);
        id
    }

    pub(super) fn push_recon(&mut self, recon: R) -> ReconId {
        let id = ReconId(u32::try_from(self.recons.len()).expect("CTU recon arena overflow"));
        self.recons.push(recon);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctu_arena_clear_drops_all_prior_entries() {
        let mut arenas = CtuArenas::<u8, u8, u8>::default();
        let _ = arenas.push_plan(1);
        let _ = arenas.push_coeffs(2);
        let _ = arenas.push_recon(3);
        arenas.clear();
        assert_eq!(arenas.mark(), ArenaMark::default());
    }
}
