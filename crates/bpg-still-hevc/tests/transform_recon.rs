//! Verify the encoder's reconstruction path (dequant + inverse transform) is
//! bit-identical to `bpg-hevc-decode`'s. This is the invariant that lets the
//! encoder feed decoder-accurate reconstructed neighbours into intra
//! prediction: for any coded levels, `bpg_still_hevc::transform` and
//! `bpg_hevc_decode::hevc::transform` must produce the same residual.

use bpg_hevc_decode::hevc::transform as dec_tf;
use bpg_still_hevc::transform as enc_tf;

/// Deterministic pseudo-random level block with a realistic sparse-ish shape.
fn make_levels(n: usize, seed: u64) -> Vec<i16> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as i64
    };
    (0..n)
        .map(|i| {
            // energy concentrated toward low frequencies, with occasional spikes
            let mag = if i < n / 4 {
                (next() % 60) - 30
            } else {
                (next() % 9) - 4
            };
            mag as i16
        })
        .collect()
}

fn check(log2_size: u8, qp: i32, bit_depth: u8, is_intra_4x4_luma: bool, seed: u64) {
    let n = 1usize << (2 * log2_size);
    let levels = make_levels(n, seed);

    // Encoder path.
    let enc_res =
        enc_tf::reconstruct_residual(&levels, log2_size, qp, bit_depth, is_intra_4x4_luma);

    // Decoder path: dequantize in place, then inverse transform.
    let mut coeffs = levels.clone();
    dec_tf::dequantize(
        &mut coeffs,
        dec_tf::DequantParams {
            qp,
            bit_depth,
            log2_tr_size: log2_size,
        },
    );
    let mut dec_res = vec![0i16; n];
    dec_tf::inverse_transform(
        &coeffs,
        &mut dec_res,
        1 << log2_size,
        bit_depth,
        is_intra_4x4_luma,
    );

    assert_eq!(
        enc_res, dec_res,
        "reconstruction mismatch: log2_size={log2_size} qp={qp} bd={bit_depth} dst={is_intra_4x4_luma}"
    );
}

#[test]
fn dequant_matches_decoder() {
    // Compare just the dequant step across sizes/QPs.
    for log2_size in 2..=5u8 {
        let n = 1usize << (2 * log2_size);
        for qp in [4, 17, 22, 28, 32, 37, 44, 51] {
            let levels = make_levels(n, qp as u64 * 7 + log2_size as u64);
            let mut a = levels.clone();
            enc_tf::dequantize(&mut a, log2_size, qp, 8);
            let mut b = levels.clone();
            dec_tf::dequantize(
                &mut b,
                dec_tf::DequantParams {
                    qp,
                    bit_depth: 8,
                    log2_tr_size: log2_size,
                },
            );
            assert_eq!(a, b, "dequant mismatch log2={log2_size} qp={qp}");
        }
    }
}

#[test]
fn reconstruction_matches_decoder_8bit() {
    for log2_size in 2..=5u8 {
        for qp in [10, 22, 28, 32, 37, 44] {
            check(log2_size, qp, 8, false, qp as u64 + log2_size as u64 * 101);
        }
    }
}

#[test]
fn reconstruction_matches_decoder_dst_4x4() {
    for qp in [10, 22, 28, 32, 37, 44] {
        check(2, qp, 8, true, qp as u64 * 13 + 5);
    }
}
