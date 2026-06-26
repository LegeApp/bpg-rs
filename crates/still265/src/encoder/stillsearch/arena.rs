//! CTU-local coefficient arena.
//!
//! Quantized levels for every evaluated transform block (winners and losers)
//! are appended to one flat CTU-local buffer; callers hold small `Copy`
//! [`CoeffId`] handles instead of per-block `Vec<i16>`s. Loser candidates are
//! discarded simply by dropping their plan/IDs — the backing storage is
//! reclaimed in bulk at the next [`CoeffArena::clear`] (one per CTU), so handles
//! stay valid for the whole CTU search and through final emit.

/// Handle into a [`CoeffArena`]. Stable for the lifetime of one CTU search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CoeffId(u32);

#[derive(Default, Debug)]
pub(super) struct CoeffArena {
    data: Vec<i16>,
    spans: Vec<(u32, u32)>,
}

impl CoeffArena {
    /// Drop every block. Backing capacity is retained for reuse next CTU.
    pub(super) fn clear(&mut self) {
        self.data.clear();
        self.spans.clear();
    }

    /// Append a block's levels, returning its handle.
    pub(super) fn push(&mut self, levels: &[i16]) -> CoeffId {
        let start = u32::try_from(self.data.len()).expect("CTU coeff arena overflow");
        self.data.extend_from_slice(levels);
        let id = CoeffId(u32::try_from(self.spans.len()).expect("CTU coeff arena overflow"));
        let len = u32::try_from(levels.len()).expect("coeff block too large");
        self.spans.push((start, len));
        id
    }

    /// Borrow a previously-pushed block's levels.
    pub(super) fn get(&self, id: CoeffId) -> &[i16] {
        let (start, len) = self.spans[id.0 as usize];
        &self.data[start as usize..(start + len) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_roundtrips_blocks_and_clears() {
        let mut arena = CoeffArena::default();
        let a = arena.push(&[1, 2, 3]);
        let b = arena.push(&[]);
        let c = arena.push(&[4, 5]);
        assert_eq!(arena.get(a), &[1, 2, 3]);
        assert_eq!(arena.get(b), &[] as &[i16]);
        assert_eq!(arena.get(c), &[4, 5]);
        arena.clear();
        let d = arena.push(&[9]);
        assert_eq!(arena.get(d), &[9]);
    }
}
