//! HEVC tile geometry, shared by the decoder (tile-scan order, cross-tile
//! availability) and the still265 encoder (which depends on this crate).
//!
//! Builds the tile boundary tables and the CTB raster<->tile-scan address
//! mappings defined in H.265 clause 6.5, plus the cross-tile availability
//! predicate. Tiles are always CTB-aligned, so two sample positions share a
//! tile iff their containing CTBs share a tile.
//!
//! Reference: `libbpg/libavcodec/hevc_ps.c` (col_bd/row_bd, ctb_addr_rs_to_ts,
//! tile_id) and `hevc.c:2343-2346` (boundary availability flags).

use super::params::TileInfo;

/// Uniform tile sizes (CTB units) along one axis: `n` tiles spanning `total`
/// CTBs, per the H.265 6.5.1 uniform-spacing derivation. The single source of
/// truth for [`TileGrid::uniform`] and the uniform branch of
/// [`TileGrid::from_tile_info`].
#[inline]
pub fn uniform_spacing(total: u32, n: u32) -> Vec<u32> {
    (0..n)
        .map(|i| (i + 1) * total / n - i * total / n)
        .collect()
}

/// Plane subsampling shifts `(x, y)` for component `c_idx` under the given
/// `chroma_array_type` (H.265 SPS semantics): luma `(0,0)`, 4:2:0 chroma
/// `(1,1)`, 4:2:2 chroma `(1,0)`, 4:4:4 chroma `(0,0)`. Shared by the decoder
/// and encoder so the chroma-format mapping cannot drift between them.
#[inline]
pub fn plane_shifts(c_idx: u8, chroma_array_type: u8) -> (u8, u8) {
    if c_idx == 0 {
        return (0, 0);
    }
    match chroma_array_type {
        1 => (1, 1),
        2 => (1, 0),
        _ => (0, 0),
    }
}

/// Tile partition geometry for one picture, in CTB units.
#[derive(Debug, Clone)]
pub struct TileGrid {
    /// Picture width/height in CTBs.
    pub ctb_width: u32,
    pub ctb_height: u32,
    /// Column boundaries in CTBs: `col_bd[0]=0 .. col_bd[num_cols]=ctb_width`.
    pub col_bd: Vec<u32>,
    /// Row boundaries in CTBs: `row_bd[0]=0 .. row_bd[num_rows]=ctb_height`.
    pub row_bd: Vec<u32>,
    /// Tile-scan -> raster CTB address (length `ctb_width*ctb_height`).
    pub ctb_addr_ts_to_rs: Vec<u32>,
    /// Raster -> tile-scan CTB address (length `ctb_width*ctb_height`).
    pub ctb_addr_rs_to_ts: Vec<u32>,
}

impl TileGrid {
    /// Trivial single-tile grid: tile-scan == raster scan. Used when tiles are
    /// disabled so callers can use one code path.
    pub fn single(ctb_width: u32, ctb_height: u32) -> Self {
        let n = (ctb_width * ctb_height) as usize;
        let identity: Vec<u32> = (0..n as u32).collect();
        Self {
            ctb_width,
            ctb_height,
            col_bd: vec![0, ctb_width],
            row_bd: vec![0, ctb_height],
            ctb_addr_ts_to_rs: identity.clone(),
            ctb_addr_rs_to_ts: identity,
        }
    }

    /// Build the grid from parsed PPS `TileInfo`. `column_widths`/`row_heights`
    /// in `TileInfo` are in CTBs and cover all but the last tile (the remainder
    /// fills the last); for uniform spacing they are derived per H.265 6.5.1.
    pub fn from_tile_info(ctb_width: u32, ctb_height: u32, ti: &TileInfo) -> Self {
        let num_cols = ti.num_tile_columns_minus1 as u32 + 1;
        let num_rows = ti.num_tile_rows_minus1 as u32 + 1;

        let column_width: Vec<u32> = if ti.uniform_spacing_flag {
            uniform_spacing(ctb_width, num_cols)
        } else {
            let mut w: Vec<u32> = ti.column_widths.iter().map(|&v| v as u32).collect();
            let sum: u32 = w.iter().sum();
            w.push(ctb_width - sum);
            w
        };
        let row_height: Vec<u32> = if ti.uniform_spacing_flag {
            uniform_spacing(ctb_height, num_rows)
        } else {
            let mut h: Vec<u32> = ti.row_heights.iter().map(|&v| v as u32).collect();
            let sum: u32 = h.iter().sum();
            h.push(ctb_height - sum);
            h
        };

        Self::from_sizes(ctb_width, ctb_height, &column_width, &row_height)
    }

    /// Build a uniform `num_cols x num_rows` grid over a picture of
    /// `ctb_width x ctb_height` CTBs, applying the H.265 6.5.1 uniform-spacing
    /// derivation. This is the single source of truth for uniform tile sizing,
    /// shared with the `uniform_spacing_flag=1` branch of [`from_tile_info`] so
    /// the encoder's partition and the decoder's derivation cannot drift.
    pub fn uniform(ctb_width: u32, ctb_height: u32, num_cols: u32, num_rows: u32) -> Self {
        let column_width = uniform_spacing(ctb_width, num_cols);
        let row_height = uniform_spacing(ctb_height, num_rows);
        Self::from_sizes(ctb_width, ctb_height, &column_width, &row_height)
    }

    /// Build the grid directly from per-tile column widths and row heights
    /// (CTB units). Used by the encoder, which decides the partition itself.
    pub fn from_sizes(
        ctb_width: u32,
        ctb_height: u32,
        column_width: &[u32],
        row_height: &[u32],
    ) -> Self {
        let num_cols = column_width.len() as u32;
        let num_rows = row_height.len() as u32;

        let mut col_bd = Vec::with_capacity(num_cols as usize + 1);
        col_bd.push(0);
        for &w in column_width {
            col_bd.push(col_bd.last().unwrap() + w);
        }
        let mut row_bd = Vec::with_capacity(num_rows as usize + 1);
        row_bd.push(0);
        for &h in row_height {
            row_bd.push(row_bd.last().unwrap() + h);
        }

        let area = (ctb_width * ctb_height) as usize;
        let mut ctb_addr_rs_to_ts = vec![0u32; area];
        let mut ctb_addr_ts_to_rs = vec![0u32; area];
        for ctb_addr_rs in 0..area as u32 {
            let tb_x = ctb_addr_rs % ctb_width;
            let tb_y = ctb_addr_rs / ctb_width;
            let tile_x = (0..num_cols)
                .find(|&i| tb_x < col_bd[i as usize + 1])
                .unwrap();
            let tile_y = (0..num_rows)
                .find(|&i| tb_y < row_bd[i as usize + 1])
                .unwrap();
            let mut val = 0u32;
            for i in 0..tile_x {
                val += row_height[tile_y as usize] * column_width[i as usize];
            }
            for i in 0..tile_y {
                val += ctb_width * row_height[i as usize];
            }
            val += (tb_y - row_bd[tile_y as usize]) * column_width[tile_x as usize] + tb_x
                - col_bd[tile_x as usize];
            ctb_addr_rs_to_ts[ctb_addr_rs as usize] = val;
            ctb_addr_ts_to_rs[val as usize] = ctb_addr_rs;
        }

        Self {
            ctb_width,
            ctb_height,
            col_bd,
            row_bd,
            ctb_addr_ts_to_rs,
            ctb_addr_rs_to_ts,
        }
    }

    #[inline]
    pub fn num_cols(&self) -> u32 {
        self.col_bd.len() as u32 - 1
    }
    #[inline]
    pub fn num_rows(&self) -> u32 {
        self.row_bd.len() as u32 - 1
    }
    #[inline]
    pub fn num_tiles(&self) -> u32 {
        self.num_cols() * self.num_rows()
    }
    #[inline]
    pub fn is_single(&self) -> bool {
        self.num_tiles() == 1
    }

    /// Tile column index containing CTB column `x_ctb`.
    #[inline]
    pub fn tile_col_of_ctb(&self, x_ctb: u32) -> u32 {
        (0..self.num_cols())
            .find(|&i| x_ctb < self.col_bd[i as usize + 1])
            .unwrap_or(self.num_cols() - 1)
    }
    /// Tile row index containing CTB row `y_ctb`.
    #[inline]
    pub fn tile_row_of_ctb(&self, y_ctb: u32) -> u32 {
        (0..self.num_rows())
            .find(|&i| y_ctb < self.row_bd[i as usize + 1])
            .unwrap_or(self.num_rows() - 1)
    }

    /// Linear tile id (row-major) of the tile containing CTB (x_ctb, y_ctb).
    #[inline]
    pub fn tile_id_of_ctb(&self, x_ctb: u32, y_ctb: u32) -> u32 {
        self.tile_row_of_ctb(y_ctb) * self.num_cols() + self.tile_col_of_ctb(x_ctb)
    }

    /// Whether two CTB positions lie in the same tile.
    #[inline]
    pub fn same_tile_ctb(&self, ax_ctb: u32, ay_ctb: u32, bx_ctb: u32, by_ctb: u32) -> bool {
        self.tile_col_of_ctb(ax_ctb) == self.tile_col_of_ctb(bx_ctb)
            && self.tile_row_of_ctb(ay_ctb) == self.tile_row_of_ctb(by_ctb)
    }

    /// Whether two luma *pixel* positions lie in the same tile. `ctb_log2` is the
    /// luma CTB log2. Short-circuits to `true` for a single-tile grid (the common
    /// case). Shared by the decoder and encoder so cross-tile availability is
    /// derived identically on both sides.
    #[inline]
    pub fn same_tile_px(&self, ax: u32, ay: u32, bx: u32, by: u32, ctb_log2: u8) -> bool {
        if self.is_single() {
            return true;
        }
        self.same_tile_ctb(
            ax >> ctb_log2,
            ay >> ctb_log2,
            bx >> ctb_log2,
            by >> ctb_log2,
        )
    }

    /// Plane-pixel bounds `(x0, y0, x1, y1)` (exclusive upper) of the tile
    /// containing the block at plane coords `(px, py)`. `ctb_log2` is the luma
    /// CTB log2; `shift_x`/`shift_y` are the plane's chroma subsampling shifts
    /// (0 for luma, 1 for 4:2:0 chroma in both axes, etc.). Used to build a
    /// reference-sample clamp closure: a sample outside these bounds belongs to a
    /// different tile and must be treated as unavailable.
    #[inline]
    pub fn tile_plane_bounds(
        &self,
        px: u32,
        py: u32,
        ctb_log2: u8,
        shift_x: u8,
        shift_y: u8,
    ) -> (u32, u32, u32, u32) {
        let xc = (px << shift_x) >> ctb_log2;
        let yc = (py << shift_y) >> ctb_log2;
        let col = self.tile_col_of_ctb(xc) as usize;
        let row = self.tile_row_of_ctb(yc) as usize;
        let cl = ctb_log2 - shift_x;
        let rl = ctb_log2 - shift_y;
        (
            self.col_bd[col] << cl,
            self.row_bd[row] << rl,
            self.col_bd[col + 1] << cl,
            self.row_bd[row + 1] << rl,
        )
    }

    /// CTB bounds `(x0, y0, x1, y1)` (CTB units, exclusive upper) of `tile_idx`
    /// (row-major). Used to iterate one tile's CTBs in raster order.
    pub fn tile_ctb_bounds(&self, tile_idx: u32) -> (u32, u32, u32, u32) {
        let col = tile_idx % self.num_cols();
        let row = tile_idx / self.num_cols();
        (
            self.col_bd[col as usize],
            self.row_bd[row as usize],
            self.col_bd[col as usize + 1],
            self.row_bd[row as usize + 1],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tile_is_identity() {
        let g = TileGrid::single(5, 4);
        assert_eq!(g.num_tiles(), 1);
        for rs in 0..20u32 {
            assert_eq!(g.ctb_addr_rs_to_ts[rs as usize], rs);
            assert_eq!(g.ctb_addr_ts_to_rs[rs as usize], rs);
        }
    }

    #[test]
    fn uniform_2x2_scan_and_membership() {
        // 4x4 CTBs split into 2x2 tiles -> each tile is 2x2.
        let g = TileGrid::from_sizes(4, 4, &[2, 2], &[2, 2]);
        assert_eq!(g.num_tiles(), 4);
        // Tile 0 = CTBs (0,0),(1,0),(0,1),(1,1) -> ts 0..4 in raster-within-tile.
        assert_eq!(g.ctb_addr_rs_to_ts[0], 0); // (0,0)
        assert_eq!(g.ctb_addr_rs_to_ts[1], 1); // (1,0)
        assert_eq!(g.ctb_addr_rs_to_ts[4], 2); // (0,1)
        assert_eq!(g.ctb_addr_rs_to_ts[5], 3); // (1,1)
        // (2,0) starts tile 1 -> ts 4.
        assert_eq!(g.ctb_addr_rs_to_ts[2], 4);
        // round-trip
        for rs in 0..16u32 {
            let ts = g.ctb_addr_rs_to_ts[rs as usize];
            assert_eq!(g.ctb_addr_ts_to_rs[ts as usize], rs);
        }
        // membership: (1,0) and (2,0) are in different tiles (col boundary at 2).
        assert!(!g.same_tile_ctb(1, 0, 2, 0));
        assert!(g.same_tile_ctb(0, 0, 1, 1));
        assert_eq!(g.tile_ctb_bounds(3), (2, 2, 4, 4));
    }

    #[test]
    fn uniform_spacing_uneven() {
        // 5 CTBs wide, 2 columns -> widths [2,3] per 6.5.1 derivation.
        let ti = TileInfo {
            num_tile_columns_minus1: 1,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_widths: vec![],
            row_heights: vec![],
            loop_filter_across_tiles_enabled_flag: true,
        };
        let g = TileGrid::from_tile_info(5, 1, &ti);
        assert_eq!(g.col_bd, vec![0, 2, 5]);
    }
}
