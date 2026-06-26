use bpg_hevc_decode::hevc::slice::IntraPredMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AngularExclusionMode {
    Off,
    Game,
    Iame,
    Tsame,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AngularExclusionConfig {
    pub(super) mode: AngularExclusionMode,
    pub(super) game_ref_var_threshold: f32,
    pub(super) iame_factor: f32,
    pub(super) min_angular_keep: usize,
    pub(super) min_log2_size: u8,
    pub(super) protect_mpm: bool,
}

impl Default for AngularExclusionConfig {
    fn default() -> Self {
        Self {
            mode: AngularExclusionMode::Off,
            game_ref_var_threshold: 8.0,
            iame_factor: 0.5,
            min_angular_keep: 6,
            min_log2_size: 4,
            protect_mpm: true,
        }
    }
}

impl AngularExclusionConfig {
    #[inline]
    pub(super) fn enabled(self) -> bool {
        self.mode != AngularExclusionMode::Off
    }
}

#[derive(Clone, Copy)]
pub(super) struct ModeMask {
    bits: u64,
}

impl ModeMask {
    #[inline]
    pub(super) fn all() -> Self {
        Self {
            bits: (1u64 << 35) - 1,
        }
    }

    #[inline]
    pub(super) fn insert(&mut self, mode: u8) {
        self.bits |= 1u64 << mode;
    }

    #[inline]
    fn remove(&mut self, mode: u8) {
        self.bits &= !(1u64 << mode);
    }

    #[inline]
    pub(super) fn contains(self, mode: u8) -> bool {
        mode < 64 && ((self.bits >> mode) & 1) != 0
    }

    fn protect_mpm(&mut self, mpm: [IntraPredMode; 3]) {
        for mode in mpm {
            self.insert(mode.as_u8());
        }
    }
}

pub(super) struct AngularExclusionResult {
    pub(super) mask: ModeMask,
    pub(super) game_changed: bool,
    pub(super) iame_changed: bool,
}

#[inline]
fn mean_u16(samples: &[u16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|&v| v as u64).sum::<u64>() as f32 / samples.len() as f32
}

#[inline]
fn mean_i32(samples: &[i32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|&v| v as i64).sum::<i64>() as f32 / samples.len() as f32
}

#[inline]
fn variance_i32(samples: &[i32], mean: f32) -> f32 {
    if samples.len() <= 1 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for &sample in samples {
        let d = sample as f32 - mean;
        acc += d * d;
    }
    acc / samples.len() as f32
}

fn apply_game(
    cfg: AngularExclusionConfig,
    refs: &[i32],
    center: usize,
    log2_size: u8,
    mask: &mut ModeMask,
) -> bool {
    let span = 2usize << log2_size;
    let lo = center.saturating_sub(span);
    let hi = (center + span).min(refs.len().saturating_sub(1));
    let active_refs = &refs[lo..=hi];
    let mean = mean_i32(active_refs);
    let var = variance_i32(active_refs, mean);
    if var > cfg.game_ref_var_threshold {
        return false;
    }
    for mode in 2u8..=34 {
        mask.remove(mode);
    }
    true
}

const MODE_HOR: i32 = 10;
const MODE_VER: i32 = 26;
const ANG_TABLE: [i32; 9] = [0, 2, 5, 9, 13, 17, 21, 26, 32];

#[inline]
fn intra_pred_angle(mode: u8) -> i32 {
    let m = mode as i32;
    if m >= 18 {
        let idx = (m - MODE_VER).unsigned_abs() as usize;
        let sign = if m >= MODE_VER { 1 } else { -1 };
        sign * ANG_TABLE[idx.min(8)]
    } else {
        let idx = (m - MODE_HOR).unsigned_abs() as usize;
        let sign = if m >= MODE_HOR { 1 } else { -1 };
        sign * ANG_TABLE[idx.min(8)]
    }
}

#[inline]
fn border_at(refs: &[i32], center: usize, rel: i32) -> i32 {
    let i = (center as i32 + rel).clamp(0, refs.len().saturating_sub(1) as i32) as usize;
    refs[i]
}

fn weighted_ref_mean_for_mode(
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    mode: u8,
    log2_size: u8,
) -> f32 {
    let n = 1usize << log2_size;
    let angle = intra_pred_angle(mode);
    let mut sum = 0u64;
    let mut count = 0u64;

    if mode >= 18 {
        for y in 0..n {
            for x in 0..n {
                let idx = x as i32 + 1 + (((y as i32 + 1) * angle) >> 5);
                let sample = border_at(filtered, center, idx);
                sum += sample.max(0) as u64;
                count += 1;
            }
        }
    } else {
        for y in 0..n {
            for x in 0..n {
                let idx = y as i32 + 1 + (((x as i32 + 1) * angle) >> 5);
                let sample = if idx >= 0 {
                    border_at(unfiltered, center, -idx)
                } else {
                    border_at(filtered, center, -idx)
                };
                sum += sample.max(0) as u64;
                count += 1;
            }
        }
    }

    if count == 0 {
        border_at(unfiltered, center, 0) as f32
    } else {
        sum as f32 / count as f32
    }
}

fn apply_iame(
    cfg: AngularExclusionConfig,
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    src: &[u16],
    log2_size: u8,
    mask: &mut ModeMask,
) -> bool {
    let pu_mean = mean_u16(src);
    let mut diffs = [(0u8, 0.0f32); 33];
    let mut dmax = 0.0f32;

    for (slot, mode) in (2u8..=34).enumerate() {
        let ref_mean = weighted_ref_mean_for_mode(unfiltered, filtered, center, mode, log2_size);
        let diff = (pu_mean - ref_mean).abs();
        diffs[slot] = (mode, diff);
        dmax = dmax.max(diff);
    }

    if dmax <= f32::EPSILON {
        return false;
    }

    let threshold = cfg.iame_factor * dmax;
    let mut changed = false;
    for &(mode, diff) in &diffs {
        if diff >= threshold {
            mask.remove(mode);
            changed = true;
        }
    }

    let kept = (2u8..=34).filter(|&mode| mask.contains(mode)).count();
    if kept < cfg.min_angular_keep {
        let mut sorted = diffs;
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        for &(mode, _) in sorted.iter().take(cfg.min_angular_keep) {
            mask.insert(mode);
        }
    }

    changed
}

pub(super) fn angular_exclusion_mask(
    cfg: AngularExclusionConfig,
    src: &[u16],
    unfiltered_refs: &[i32],
    filtered_refs: &[i32],
    center: usize,
    log2_size: u8,
    mpm: [IntraPredMode; 3],
) -> AngularExclusionResult {
    let mut mask = ModeMask::all();
    if !cfg.enabled() || log2_size < cfg.min_log2_size {
        return AngularExclusionResult {
            mask,
            game_changed: false,
            iame_changed: false,
        };
    }

    let mut game_changed = false;
    let mut iame_changed = false;
    match cfg.mode {
        AngularExclusionMode::Off => {}
        AngularExclusionMode::Game => {
            game_changed = apply_game(cfg, unfiltered_refs, center, log2_size, &mut mask);
        }
        AngularExclusionMode::Iame => {
            iame_changed = apply_iame(
                cfg,
                unfiltered_refs,
                filtered_refs,
                center,
                src,
                log2_size,
                &mut mask,
            );
        }
        AngularExclusionMode::Tsame => {
            game_changed = apply_game(cfg, unfiltered_refs, center, log2_size, &mut mask);
            if (2u8..=34).any(|mode| mask.contains(mode)) {
                iame_changed = apply_iame(
                    cfg,
                    unfiltered_refs,
                    filtered_refs,
                    center,
                    src,
                    log2_size,
                    &mut mask,
                );
            }
        }
    }

    mask.insert(0);
    mask.insert(1);
    if cfg.protect_mpm {
        mask.protect_mpm(mpm);
    }

    AngularExclusionResult {
        mask,
        game_changed,
        iame_changed,
    }
}
