//! Bit-level I/O primitives used by the BPG container format and the
//! "modified HEVC" bitstream rewriting: big-endian base-128 "ue7" varints
//! (used for container header fields) and MSB-first bit reading/writing with
//! Exp-Golomb (`ue(v)`) coding (used for HEVC RBSP rewriting).
//!
//! Ported from `put_ue`/`get_ue`/`get_bits`/`put_bits`/`get_ue_golomb`/
//! `put_ue_golomb` in `libbpg-0.9.8/bpgenc.c`.

/// Encode `v` as a big-endian base-128 varint ("ue7"): each byte holds 7 bits
/// of the value, with the high bit set on every byte except the last.
pub fn write_ue7(out: &mut Vec<u8>, v: u32) {
    let mut i = 1u32;
    while i < 5 {
        if v < (1u32 << (7 * i)) {
            break;
        }
        i += 1;
    }
    let mut j = i as i32 - 1;
    while j >= 1 {
        out.push((((v >> (7 * j)) & 0x7f) as u8) | 0x80);
        j -= 1;
    }
    out.push((v & 0x7f) as u8);
}

/// Decode a "ue7" varint starting at `*pos`, advancing `*pos` past it.
/// Returns `None` if the buffer ends before a terminating byte is found.
pub fn read_ue7(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let mut v: u32 = 0;
    loop {
        let b = *buf.get(*pos)?;
        *pos += 1;
        v = (v << 7) | (b & 0x7f) as u32;
        if b & 0x80 == 0 {
            break;
        }
    }
    Some(v)
}

/// MSB-first bit reader over a byte slice. Reading past the end of the
/// buffer yields zero bits, matching `get_bits` in `bpgenc.c`.
pub struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Current position in bits from the start of the buffer.
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    pub fn skip_bits(&mut self, n: usize) {
        self.bit_pos += n;
    }

    /// Read `n` bits (1 <= n <= 25), MSB first.
    pub fn read_bits(&mut self, n: u32) -> u32 {
        let p = self.bit_pos >> 3;
        let v: u32 = if p + 3 < self.buf.len() {
            ((self.buf[p] as u32) << 24)
                | ((self.buf[p + 1] as u32) << 16)
                | ((self.buf[p + 2] as u32) << 8)
                | (self.buf[p + 3] as u32)
        } else {
            let mut v = 0u32;
            for i in 0..3usize {
                if p + i < self.buf.len() {
                    v |= (self.buf[p + i] as u32) << (24 - i * 8);
                }
            }
            v
        };
        let shift = 32 - (self.bit_pos & 7) as u32 - n;
        let result = (v >> shift) & ((1u32 << n) - 1);
        self.bit_pos += n as usize;
        result
    }

    /// Read `n` bits (1 <= n <= 32), MSB first.
    pub fn read_bits_long(&mut self, n: u32) -> u32 {
        if n <= 25 {
            self.read_bits(n)
        } else {
            let n2 = n - 16;
            let hi = self.read_bits(16) << n2;
            hi | self.read_bits(n2)
        }
    }

    /// Read an Exp-Golomb `ue(v)` value. Returns `None` if more than 32
    /// leading zero bits are encountered (matches the `0xffffffff` error
    /// sentinel in `get_ue_golomb`).
    pub fn read_ue_golomb(&mut self) -> Option<u32> {
        let mut i = 0u32;
        loop {
            if self.read_bits(1) != 0 {
                break;
            }
            i += 1;
            if i == 32 {
                return None;
            }
        }
        if i == 0 {
            Some(0)
        } else {
            Some(((1u32 << i) | self.read_bits_long(i)) - 1)
        }
    }

    /// Read a signed Exp-Golomb `se(v)` value, per H.265 9.2.2: `codeNum =
    /// ue(v)`; `se(v) = (-1)^(codeNum+1) * ceil(codeNum / 2)`, i.e. codeNum
    /// 0,1,2,3,4 -> 0,1,-1,2,-2. Mirrors `BitstreamReader::read_se` in
    /// `bpg-hevc-decode`.
    pub fn read_se_golomb(&mut self) -> Option<i32> {
        let ue = self.read_ue_golomb()?;
        let value = ue.div_ceil(2) as i32;
        Some(if ue & 1 == 1 { value } else { -value })
    }
}

/// MSB-first bit writer, growing a byte buffer as bits are written.
pub struct BitWriter {
    buf: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
        }
    }

    pub fn write_bit(&mut self, bit: u32) {
        let byte_idx = self.bit_pos >> 3;
        if byte_idx >= self.buf.len() {
            self.buf.push(0);
        }
        self.buf[byte_idx] |= ((bit & 1) as u8) << (7 - (self.bit_pos & 7));
        self.bit_pos += 1;
    }

    /// Write the `n` low bits of `v`, MSB first.
    pub fn write_bits(&mut self, n: u32, v: u32) {
        for i in 0..n {
            self.write_bit((v >> (n - 1 - i)) & 1);
        }
    }

    /// Write an Exp-Golomb `ue(v)` value.
    pub fn write_ue_golomb(&mut self, v: u32) {
        let v = v + 1;
        let mut n = 0u32;
        let mut a = v;
        while a != 0 {
            a >>= 1;
            n += 1;
        }
        if n > 1 {
            self.write_bits(n - 1, 0);
        }
        self.write_bits(n, v);
    }

    /// Write a signed Exp-Golomb `se(v)` value: maps `v` to `codeNum` per
    /// H.265 9.2.2 (0,1,-1,2,-2 -> codeNum 0,1,2,3,4, the inverse of
    /// `read_se_golomb`) and writes `ue(codeNum)`.
    pub fn write_se_golomb(&mut self, v: i32) {
        let code_num = if v <= 0 {
            (-2 * v) as u32
        } else {
            (2 * v - 1) as u32
        };
        self.write_ue_golomb(code_num);
    }

    /// Pad with zero bits up to the next byte boundary.
    pub fn byte_align(&mut self) {
        let rem = self.bit_pos & 7;
        if rem != 0 {
            self.write_bits((8 - rem) as u32, 0);
        }
    }

    /// Current position in bits from the start of the buffer.
    pub fn bit_len(&self) -> usize {
        self.bit_pos
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue7_small_values() {
        let mut out = Vec::new();
        write_ue7(&mut out, 0);
        assert_eq!(out, vec![0x00]);

        let mut out = Vec::new();
        write_ue7(&mut out, 127);
        assert_eq!(out, vec![0x7f]);

        let mut out = Vec::new();
        write_ue7(&mut out, 128);
        assert_eq!(out, vec![0x81, 0x00]);

        let mut out = Vec::new();
        write_ue7(&mut out, 3200); // dusk.png width
                                   // 3200 = 0b110010000000 -> 13 bits -> 2 groups of 7
                                   // hi 7 bits: 3200 >> 7 = 25 = 0x19, lo 7 bits: 3200 & 0x7f = 0x00
        assert_eq!(out, vec![0x19 | 0x80, 0x00]);
    }

    #[test]
    fn ue7_round_trip() {
        for v in [
            0u32,
            1,
            126,
            127,
            128,
            255,
            256,
            16383,
            16384,
            2_097_151,
            2_097_152,
            u32::MAX,
        ] {
            let mut out = Vec::new();
            write_ue7(&mut out, v);
            let mut pos = 0;
            assert_eq!(read_ue7(&out, &mut pos), Some(v), "v={v}");
            assert_eq!(pos, out.len());
        }
    }

    #[test]
    fn ue7_truncated_returns_none() {
        // 0x81 alone has the continuation bit set but no terminating byte
        let mut pos = 0;
        assert_eq!(read_ue7(&[0x81], &mut pos), None);
    }

    #[test]
    fn bitreader_basic() {
        // bytes: 1011_0010 1110_0001
        let buf = [0b1011_0010, 0b1110_0001];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(1), 1);
        assert_eq!(r.read_bits(3), 0b011);
        assert_eq!(r.read_bits(4), 0b0010);
        assert_eq!(r.read_bits(8), 0b1110_0001);
    }

    #[test]
    fn bitreader_past_end_is_zero() {
        let buf = [0xffu8];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(8), 0xff);
        // past the end: zero bits
        assert_eq!(r.read_bits(8), 0);
    }

    #[test]
    fn bitwriter_matches_bitreader() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(3, 0b011);
        w.write_bits(4, 0b0010);
        w.write_bits(8, 0b1110_0001);
        w.byte_align();
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0b1011_0010, 0b1110_0001]);

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(1), 1);
        assert_eq!(r.read_bits(3), 0b011);
        assert_eq!(r.read_bits(4), 0b0010);
        assert_eq!(r.read_bits(8), 0b1110_0001);
    }

    #[test]
    fn ue_golomb_known_values() {
        // ue(v) Exp-Golomb codes: 0 -> "1", 1 -> "010", 2 -> "011",
        // 3 -> "00100", 4 -> "00101", ...
        let cases: [(u32, &str); 5] =
            [(0, "1"), (1, "010"), (2, "011"), (3, "00100"), (4, "00101")];
        for (v, bits) in cases {
            let mut w = BitWriter::new();
            w.write_ue_golomb(v);
            assert_eq!(w.bit_len(), bits.len(), "v={v}");
            let mut check = BitWriter::new();
            for c in bits.chars() {
                check.write_bit(if c == '1' { 1 } else { 0 });
            }
            assert_eq!(w.into_bytes(), check.into_bytes(), "v={v}");
        }
    }

    #[test]
    fn se_golomb_known_values() {
        // se(v) <-> codeNum: 0<->0, 1<->1, -1<->2, 2<->3, -2<->4 (H.265 Table 9-3).
        for (v, code_num) in [
            (0i32, 0u32),
            (1, 1),
            (-1, 2),
            (2, 3),
            (-2, 4),
            (3, 5),
            (-3, 6),
        ] {
            let mut w = BitWriter::new();
            w.write_se_golomb(v);
            let mut ue = BitWriter::new();
            ue.write_ue_golomb(code_num);
            assert_eq!(w.into_bytes(), ue.into_bytes(), "v={v}");
        }
    }

    #[test]
    fn se_golomb_round_trip() {
        for v in [0i32, 1, -1, 2, -2, 3, -3, 100, -100, 1 << 16, -(1 << 16)] {
            let mut w = BitWriter::new();
            w.write_se_golomb(v);
            w.byte_align();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_se_golomb(), Some(v), "v={v}");
        }
    }

    #[test]
    fn ue_golomb_round_trip() {
        for v in [0u32, 1, 2, 3, 4, 5, 100, 1000, 65535, 1 << 20] {
            let mut w = BitWriter::new();
            w.write_ue_golomb(v);
            w.byte_align();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_ue_golomb(), Some(v), "v={v}");
        }
    }
}
