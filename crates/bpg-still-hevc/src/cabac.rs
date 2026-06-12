//! CABAC (Context-Adaptive Binary Arithmetic Coding) encoder.
//!
//! Ported from `Entropy::encodeBin`/`encodeBinEP`/`encodeBinsEP`/
//! `encodeBinTrm`/`finish`/`writeOut`/`start`/`sbacInit` in
//! `x265_4.1/source/encoder/entropy.cpp`, and the `g_lpsTable` LPS-range
//! table from `x265_4.1/source/common/constants.cpp` (identical to H.265
//! Table 9-46 / the `LPS_TABLE` used by the decoder-side
//! `bpg-hevc-decode::hevc::cabac::CabacDecoder`). The state-transition tables
//! are H.265 Table 9-43, also duplicated from the decoder for a
//! self-contained encoder-side port.
//!
//! Output bits are written MSB-first into a `bpg_bitstream::BitWriter`: each
//! full byte produced by `write_out` is appended via `write_bits(8, byte)`,
//! and `finish` appends the final (possibly non-byte-aligned) tail bits. The
//! caller is responsible for `rbsp_trailing_bits()` (stop bit + zero padding)
//! after `finish`.

use bpg_bitstream::BitWriter;

/// LPS range table, H.265 Table 9-46 (`g_lpsTable` in
/// `x265_4.1/source/common/constants.cpp`).
#[rustfmt::skip]
static LPS_TABLE: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [95, 116, 137, 158],  [90, 110, 130, 150],  [85, 104, 123, 142],  [81, 99, 117, 135],
    [77, 94, 111, 128],   [73, 89, 105, 122],   [69, 85, 100, 116],   [66, 80, 95, 110],
    [62, 76, 90, 104],    [59, 72, 86, 99],     [56, 69, 81, 94],     [53, 65, 77, 89],
    [51, 62, 73, 85],     [48, 59, 69, 80],     [46, 56, 66, 76],     [43, 53, 63, 72],
    [41, 50, 59, 69],     [39, 48, 56, 65],     [37, 45, 54, 62],     [35, 43, 51, 59],
    [33, 41, 48, 56],     [32, 39, 46, 53],     [30, 37, 43, 50],     [29, 35, 41, 48],
    [27, 33, 39, 45],     [26, 31, 37, 43],     [24, 30, 35, 41],     [23, 28, 33, 39],
    [22, 27, 32, 37],     [21, 26, 30, 35],     [20, 24, 29, 33],     [19, 23, 27, 31],
    [18, 22, 26, 30],     [17, 21, 25, 28],     [16, 20, 23, 27],     [15, 19, 22, 25],
    [14, 18, 21, 24],     [14, 17, 20, 23],     [13, 16, 19, 22],     [12, 15, 18, 21],
    [12, 14, 17, 20],     [11, 14, 16, 19],     [11, 13, 15, 18],     [10, 12, 15, 17],
    [10, 12, 14, 16],     [9, 11, 13, 15],      [9, 11, 12, 14],      [8, 10, 12, 14],
    [8, 9, 11, 13],       [7, 9, 11, 12],       [7, 9, 10, 12],       [7, 8, 10, 11],
    [6, 8, 9, 11],        [6, 7, 9, 10],        [6, 7, 8, 9],         [2, 2, 2, 2],
];

/// MPS state transition, H.265 Table 9-43.
#[rustfmt::skip]
static STATE_TRANS_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// LPS state transition, H.265 Table 9-43.
#[rustfmt::skip]
static STATE_TRANS_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// A single CABAC context model: state index (0..=63) and most-probable-symbol.
#[derive(Clone, Copy, Debug)]
pub struct ContextModel {
    state: u8,
    mps: u8,
}

impl ContextModel {
    /// Initialize a context model from its H.265 Table 9-4 `init_value` and
    /// the slice QP, per H.265 9.3.2.2 / `sbacInit` in
    /// `x265_4.1/source/encoder/entropy.cpp`:
    /// `preCtxState = Clip3(1, 126, ((m * Clip3(0,51,SliceQpY)) >> 4) + n)`
    /// where `m = (init_value >> 4) * 5 - 45`, `n = ((init_value & 15) << 3) - 16`.
    pub fn new(init_value: u8, slice_qp: i32) -> Self {
        let m = (init_value >> 4) as i32 * 5 - 45;
        let n = ((init_value & 15) << 3) as i32 - 16;
        let qp = slice_qp.clamp(0, 51);
        let pre_ctx_state = (((m * qp) >> 4) + n).clamp(1, 126);
        if pre_ctx_state >= 64 {
            Self { state: (pre_ctx_state - 64) as u8, mps: 1 }
        } else {
            Self { state: (63 - pre_ctx_state) as u8, mps: 0 }
        }
    }

    /// Current (state, mps), exposed for cross-checking against the
    /// decoder-side `bpg_hevc_decode::hevc::cabac::ContextModel`.
    pub fn get_state(&self) -> (u8, u8) {
        (self.state, self.mps)
    }
}

/// CABAC arithmetic encoder, mirroring x265's `Entropy` range-coder state
/// (`m_low`/`m_range`/`m_bitsLeft`/`m_bufferedByte`/`m_numBufferedBytes`).
pub struct CabacEncoder {
    low: u32,
    range: u32,
    bits_left: i32,
    buffered_byte: u8,
    num_buffered_bytes: u32,
}

impl CabacEncoder {
    /// `Entropy::start`.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            bits_left: -12,
            buffered_byte: 0xff,
            num_buffered_bytes: 0,
        }
    }

    /// `Entropy::encodeBin`.
    pub fn encode_bin(&mut self, w: &mut BitWriter, bin_value: u8, ctx: &mut ContextModel) {
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps = LPS_TABLE[ctx.state as usize][q_range_idx] as u32;
        self.range -= lps;

        let num_bits: u32;
        if bin_value != ctx.mps {
            // LPS path: CLZ(idx, lps) -> idx = bit index of lps's MSB (lps in 1..=255).
            let idx = 7 - (lps as u8).leading_zeros();
            num_bits = if ctx.state >= 63 { 6 } else { 8 - idx };
            self.low += self.range;
            self.range = lps;
            if ctx.state == 0 {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = STATE_TRANS_LPS[ctx.state as usize];
        } else {
            ctx.state = STATE_TRANS_MPS[ctx.state as usize];
            num_bits = self.range.wrapping_sub(256) >> 31;
        }

        self.low <<= num_bits;
        self.range <<= num_bits;
        self.bits_left += num_bits as i32;

        if self.bits_left >= 0 {
            self.write_out(w);
        }
    }

    /// `Entropy::encodeBinEP`.
    pub fn encode_bin_ep(&mut self, w: &mut BitWriter, bin_value: u8) {
        self.low <<= 1;
        if bin_value != 0 {
            self.low += self.range;
        }
        self.bits_left += 1;
        if self.bits_left >= 0 {
            self.write_out(w);
        }
    }

    /// `Entropy::encodeBinsEP`.
    ///
    /// Untested until Phase 5 (residual coding is the first syntax that uses
    /// the batched bypass path); `tests/cabac_roundtrip.rs` only exercises
    /// `encode_bin_ep` one bin at a time.
    pub fn encode_bins_ep(&mut self, w: &mut BitWriter, bin_values: u32, num_bins: u32) {
        let mut bin_values = bin_values;
        let mut num_bins = num_bins;
        while num_bins > 8 {
            num_bins -= 8;
            let pattern = bin_values >> num_bins;
            self.low <<= 8;
            self.low += self.range * pattern;
            bin_values -= pattern << num_bins;
            self.bits_left += 8;
            if self.bits_left >= 0 {
                self.write_out(w);
            }
        }
        self.low <<= num_bins;
        self.low += self.range * bin_values;
        self.bits_left += num_bins as i32;
        if self.bits_left >= 0 {
            self.write_out(w);
        }
    }

    /// `Entropy::encodeBinTrm`.
    pub fn encode_bin_trm(&mut self, w: &mut BitWriter, bin_value: u8) {
        self.range -= 2;
        if bin_value != 0 {
            self.low += self.range;
            self.low <<= 7;
            self.range = 2 << 7;
            self.bits_left += 7;
        } else if self.range >= 256 {
            return;
        } else {
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left += 1;
        }
        if self.bits_left >= 0 {
            self.write_out(w);
        }
    }

    /// `Entropy::writeOut`: emit one fully-determined output byte (with carry
    /// propagation through any buffered `0xff` bytes).
    fn write_out(&mut self, w: &mut BitWriter) {
        let lead_byte = self.low >> (13 + self.bits_left) as u32;
        let low_mask = u32::MAX >> (19 - self.bits_left) as u32;
        self.bits_left -= 8;
        self.low &= low_mask;

        if lead_byte == 0xff {
            self.num_buffered_bytes += 1;
        } else {
            if self.num_buffered_bytes > 0 {
                let carry = lead_byte >> 8;
                w.write_bits(8, self.buffered_byte as u32 + carry);
                let fill = (0xffu32 + carry) & 0xff;
                for _ in 1..self.num_buffered_bytes {
                    w.write_bits(8, fill);
                }
            }
            self.num_buffered_bytes = 1;
            self.buffered_byte = lead_byte as u8;
        }
    }

    /// `Entropy::finish`: flush remaining buffered bytes and the final tail
    /// bits (`m_low >> 8`, `13 + m_bitsLeft` bits). The caller must follow
    /// this with `rbsp_trailing_bits()` to byte-align.
    pub fn finish(&mut self, w: &mut BitWriter) {
        if self.low >> (21 + self.bits_left) as u32 != 0 {
            w.write_bits(8, self.buffered_byte as u32 + 1);
            while self.num_buffered_bytes > 1 {
                w.write_bits(8, 0x00);
                self.num_buffered_bytes -= 1;
            }
            self.low -= 1 << (21 + self.bits_left) as u32;
        } else {
            if self.num_buffered_bytes > 0 {
                w.write_bits(8, self.buffered_byte as u32);
            }
            while self.num_buffered_bytes > 1 {
                w.write_bits(8, 0xff);
                self.num_buffered_bytes -= 1;
            }
        }
        let n = (13 + self.bits_left) as u32;
        w.write_bits(n, self.low >> 8);
    }
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}
