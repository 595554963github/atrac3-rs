use std::collections::BTreeMap;
use std::sync::OnceLock;

use anyhow::{Result, anyhow, ensure};

use super::{
    SAMPLES_PER_FRAME,
    sound_unit::{
        ChannelSoundUnit, CodingMode, GainBand, RawBitPayload, SpectralSubband, SpectralUnit,
        TonalCodingModeSelector,
    },
};

pub const ATRAC3_HUFF_TAB_SIZES: [u8; 7] = [9, 5, 7, 9, 15, 31, 63];
pub const ATRAC3_CLC_LENGTH_TAB: [u8; 8] = [0, 4, 3, 3, 4, 4, 5, 6];
pub const ATRAC3_MANTISSA_CLC_TAB: [i8; 4] = [0, 1, -2, -1];
pub const ATRAC3_MANTISSA_VLC_TAB: [i8; 18] =
    [0, 0, 0, 1, 0, -1, 1, 0, -1, 0, 1, 1, 1, -1, -1, 1, -1, -1];
pub const ATRAC3_INV_MAX_QUANT: [f32; 8] = [
    0.0,
    1.0 / 1.5,
    1.0 / 2.5,
    1.0 / 3.5,
    1.0 / 4.5,
    1.0 / 7.5,
    1.0 / 15.5,
    1.0 / 31.5,
];
pub const ATRAC3_SUBBAND_TAB: [usize; 33] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 288, 320,
    352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 896, 1024,
];

// ---------- Allocator tuning constants ----------

/// Minimum cumulative sfIndex for a band to stay active. Subbands below
/// this threshold get the skip table index (table 0) and cost zero bits.
const ENERGY_THRESHOLD: [i32; 32] = [
    7, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 4, 5, 5, 5, 6,
    6, 7, 7, 8, 10, 13, 17, 22, 28, 35, 49, 74, 109, 155, 250, 441,
];

/// Codebook boundary used for both null-demotion and promotion decisions.
/// null_boundary(tblIdx) = CODEBOOK_BOUNDARY[tblIdx - 1]
/// promo_boundary(tblIdx) = CODEBOOK_BOUNDARY[tblIdx]
const CODEBOOK_BOUNDARY: [i32; 8] = [1, 2, 2, 2, 4, 6, 6, 40];

/// Cost reduction per below-threshold group of 4 coefficients.
const COST_REDUCTION: [i32; 8] = [6, 40, 40, 60, 76, 60, 60, 100];

/// Base cost scalar per tblIndex (0 through 7).
const BASE_COST_SCALAR: [i32; 8] = [100, 15, 20, 25, 29, 35, 45, 55];

/// sfIndex threshold offset per tblIndex (0 through 7).
const SF_THRESHOLD_OFFSET: [i32; 8] = [55, 3, 5, 7, 9, 12, 15, 18];

const ATRAC3_HUFF_TABS: [(u8, u8); 139] = [
    (31, 1),
    (32, 3),
    (33, 3),
    (34, 4),
    (35, 4),
    (36, 5),
    (37, 5),
    (38, 5),
    (39, 5),
    (31, 1),
    (32, 3),
    (30, 3),
    (33, 3),
    (29, 3),
    (31, 1),
    (32, 3),
    (30, 3),
    (33, 4),
    (29, 4),
    (34, 4),
    (28, 4),
    (31, 1),
    (32, 3),
    (30, 3),
    (33, 4),
    (29, 4),
    (34, 5),
    (28, 5),
    (35, 5),
    (27, 5),
    (31, 2),
    (32, 3),
    (30, 3),
    (33, 4),
    (29, 4),
    (34, 4),
    (28, 4),
    (38, 4),
    (24, 4),
    (35, 5),
    (27, 5),
    (36, 6),
    (26, 6),
    (37, 6),
    (25, 6),
    (31, 3),
    (32, 4),
    (30, 4),
    (33, 4),
    (29, 4),
    (34, 4),
    (28, 4),
    (46, 4),
    (16, 4),
    (35, 5),
    (27, 5),
    (36, 5),
    (26, 5),
    (37, 5),
    (25, 5),
    (38, 6),
    (24, 6),
    (39, 6),
    (23, 6),
    (40, 6),
    (22, 6),
    (41, 6),
    (21, 6),
    (42, 7),
    (20, 7),
    (43, 7),
    (19, 7),
    (44, 7),
    (18, 7),
    (45, 7),
    (17, 7),
    (31, 3),
    (62, 4),
    (0, 4),
    (32, 5),
    (30, 5),
    (33, 5),
    (29, 5),
    (34, 5),
    (28, 5),
    (35, 5),
    (27, 5),
    (36, 5),
    (26, 5),
    (37, 6),
    (25, 6),
    (38, 6),
    (24, 6),
    (39, 6),
    (23, 6),
    (40, 6),
    (22, 6),
    (41, 6),
    (21, 6),
    (42, 6),
    (20, 6),
    (43, 6),
    (19, 6),
    (44, 6),
    (18, 6),
    (45, 7),
    (17, 7),
    (46, 7),
    (16, 7),
    (47, 7),
    (15, 7),
    (48, 7),
    (14, 7),
    (49, 7),
    (13, 7),
    (50, 7),
    (12, 7),
    (51, 7),
    (11, 7),
    (52, 8),
    (10, 8),
    (53, 8),
    (9, 8),
    (54, 8),
    (8, 8),
    (55, 8),
    (7, 8),
    (56, 8),
    (6, 8),
    (57, 8),
    (5, 8),
    (58, 8),
    (4, 8),
    (59, 8),
    (3, 8),
    (60, 8),
    (2, 8),
    (61, 8),
    (1, 8),
];

const ATRAC3_MANTISSA_VLC_PAIRS: [[i8; 2]; 9] = [
    [0, 0],
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuffmanEntry {
    symbol: i8,
    code: u32,
    bits: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedSubband {
    pub table_index: u8,
    pub scale_factor_index: Option<u8>,
    pub mantissas: Vec<i8>,
    pub payload_bits: usize,
    pub mse: f32,
    pub max_abs_err: f32,  // Sony's selection criterion (max worst-case bin)
}

impl QuantizedSubband {
    pub fn uncoded(coefficients: &[f32]) -> Self {
        let max_abs = coefficients.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        Self {
            table_index: 0,
            scale_factor_index: None,
            mantissas: Vec::new(),
            payload_bits: 0,
            mse: mean_square(coefficients, &vec![0.0; coefficients.len()]),
            max_abs_err: max_abs,
        }
    }

    pub fn payload(&self, coding_mode: CodingMode) -> Result<RawBitPayload> {
        if self.table_index == 0 {
            return Ok(RawBitPayload::default());
        }
        encode_mantissas(self.table_index, coding_mode, &self.mantissas)
    }

    pub fn spectral_subband(&self, coding_mode: CodingMode) -> Result<SpectralSubband> {
        Ok(SpectralSubband {
            table_index: self.table_index,
            scale_factor_index: self.scale_factor_index,
            payload: self.payload(coding_mode)?,
        })
    }

    pub fn dequantized(&self, expected_len: usize) -> Result<Vec<f32>> {
        if self.table_index == 0 {
            return Ok(vec![0.0; expected_len]);
        }

        let scale_factor_index = self
            .scale_factor_index
            .ok_or_else(|| anyhow!("coded subband is missing a scale factor index"))?;
        ensure!(
            self.mantissas.len() == expected_len,
            "mantissa count {} does not match expected coefficient count {}",
            self.mantissas.len(),
            expected_len
        );
        let scale =
            scale_factor(scale_factor_index) * ATRAC3_INV_MAX_QUANT[self.table_index as usize];
        Ok(self
            .mantissas
            .iter()
            .map(|&mantissa| mantissa as f32 * scale)
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumEncoding {
    pub spectral_unit: SpectralUnit,
    pub quantized_subbands: Vec<QuantizedSubband>,
    pub reconstructed: Vec<f32>,
    pub payload_bits: usize,
    pub mse: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOptions {
    pub lambda: f32,
    pub target_bits: Option<usize>,
    pub max_candidates_per_band: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            lambda: 0.0001,
            target_bits: None,
            max_candidates_per_band: 64,
        }
    }
}

/// Fast peak-to-sfIndex mapper.
/// Uses IEEE 754 bit manipulation: sfIndex = 3 * exponent - 364 ± tercile correction.
/// Each sfIndex step corresponds to 2^(1/3) ≈ 1.26x amplitude.
fn fast_peak_to_sf_index(coefficients: &[f32]) -> u8 {
    let mut max_abs_bits: u32 = 0;
    for &c in coefficients {
        // |c| * 2 as unsigned removes sign bit (classic IEEE trick)
        let doubled = (c.to_bits() << 1) | 0; // shift away sign
        if doubled > max_abs_bits {
            max_abs_bits = doubled;
        }
    }
    if max_abs_bits == 0 {
        return 0;
    }
    let exponent = (max_abs_bits >> 24) as i32; // 8-bit biased exponent of doubled value
    let mantissa = max_abs_bits & 0x00FF_FFFF; // 24-bit fraction

    let mut sf_index = 3 * exponent - 364;

    // Tercile correction within each octave
    if mantissa > 0x0096_5FE9 {
        // > 2^(2/3) of mantissa range
        sf_index += 1;
    } else if mantissa < 0x0042_8A30 {
        // < 2^(1/3) of mantissa range
        sf_index -= 1;
    }

    if sf_index < 0 || sf_index > 63 {
        0
    } else {
        sf_index as u8
    }
}

/// Compute sfIndex for groups of 4 coefficients across a subband.
fn compute_group_sf_indices(coefficients: &[f32]) -> Vec<u8> {
    coefficients
        .chunks(4)
        .map(|chunk| fast_peak_to_sf_index(chunk))
        .collect()
}

/// Fast per-band bit cost estimator.
/// Estimates cost without actually encoding; uses table lookups only.
fn estimate_band_bit_cost(
    group_sf_indices: &[u8],
    tbl_index: u8,
    sf_index: i32,
    subband_width: usize,
) -> i32 {
    if tbl_index == 0 {
        return 0;
    }
    let ti = tbl_index as usize;
    let threshold = sf_index - SF_THRESHOLD_OFFSET[ti];
    let mut cost = BASE_COST_SCALAR[ti] * subband_width as i32 + 60;

    for &group_sf in group_sf_indices {
        if (group_sf as i32) < threshold {
            cost -= COST_REDUCTION[ti];
        }
    }
    cost.max(0)
}

pub fn clc_bit_width(selector: u8) -> Option<u8> {
    ATRAC3_CLC_LENGTH_TAB.get(selector as usize).copied()
}

pub fn huff_table_size(selector: u8) -> Option<u8> {
    ATRAC3_HUFF_TAB_SIZES
        .get(selector.checked_sub(1)? as usize)
        .copied()
}

pub fn inv_max_quant(selector: u8) -> Option<f32> {
    ATRAC3_INV_MAX_QUANT.get(selector as usize).copied()
}

pub fn scale_factor(index: u8) -> f32 {
    2.0f32.powf((index as f32 - 15.0) / 3.0)
}

pub fn encode_mantissas(
    selector: u8,
    coding_mode: CodingMode,
    mantissas: &[i8],
) -> Result<RawBitPayload> {
    ensure!(
        (1..=7).contains(&selector),
        "selector {} out of range",
        selector
    );
    let mut payload = RawBitPayload::default();

    match (coding_mode, selector) {
        (CodingMode::Clc, 1) => {
            ensure!(
                mantissas.len() % 2 == 0,
                "selector 1 requires an even mantissa count"
            );
            for pair in mantissas.chunks_exact(2) {
                let hi = clc_symbol_index(pair[0])?;
                let lo = clc_symbol_index(pair[1])?;
                payload.push_bits(((hi << 2) | lo) as u32, ATRAC3_CLC_LENGTH_TAB[1])?;
            }
        }
        (CodingMode::Clc, _) => {
            let width = ATRAC3_CLC_LENGTH_TAB[selector as usize];
            for &mantissa in mantissas {
                payload.push_bits(twos_complement_bits(mantissa as i32, width), width)?;
            }
        }
        (CodingMode::Vlc, 1) => {
            ensure!(
                mantissas.len() % 2 == 0,
                "selector 1 requires an even mantissa count"
            );
            for pair in mantissas.chunks_exact(2) {
                let symbol = vlc_pair_symbol(pair[0], pair[1])?;
                let entry = find_huffman_entry(selector, symbol)?;
                payload.push_bits(entry.code, entry.bits)?;
            }
        }
        (CodingMode::Vlc, _) => {
            for &mantissa in mantissas {
                let entry = find_huffman_entry(selector, mantissa)?;
                payload.push_bits(entry.code, entry.bits)?;
            }
        }
    }

    Ok(payload)
}

pub fn choose_subband_encoding(
    coefficients: &[f32],
    coding_mode: CodingMode,
    options: SearchOptions,
) -> Result<QuantizedSubband> {
    ensure!(
        !coefficients.is_empty(),
        "subband coefficients must not be empty"
    );

    let mut best = QuantizedSubband::uncoded(coefficients);
    let mut best_score = best.mse;

    for selector in 1..=7u8 {
        for scale_factor_index in 0..64u8 {
            let candidate =
                quantize_subband(coefficients, selector, scale_factor_index, coding_mode)?;
            let score = candidate.mse + options.lambda * candidate.payload_bits as f32;
            if score + 1e-12 < best_score
                || ((score - best_score).abs() <= 1e-12
                    && candidate.payload_bits < best.payload_bits)
            {
                best = candidate;
                best_score = score;
            }
        }
    }

    Ok(best)
}

pub fn build_spectral_unit(
    coefficients: &[f32],
    coding_mode: CodingMode,
    options: SearchOptions,
) -> Result<SpectrumEncoding> {
    if let Some(target_bits) = options.target_bits {
        return build_spectral_unit_budgeted(coefficients, coding_mode, options, target_bits);
    }

    ensure!(
        coefficients.len() == SAMPLES_PER_FRAME,
        "expected {} coefficients, got {}",
        SAMPLES_PER_FRAME,
        coefficients.len()
    );

    let mut quantized_subbands = Vec::with_capacity(32);
    let mut reconstructed = vec![0.0f32; coefficients.len()];

    for band in 0..32 {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        let quantized = choose_subband_encoding(&coefficients[start..end], coding_mode, options)?;
        let band_reconstructed = quantized.dequantized(end - start)?;
        reconstructed[start..end].copy_from_slice(&band_reconstructed);
        quantized_subbands.push(quantized);
    }

    let last_coded = quantized_subbands
        .iter()
        .rposition(|subband| subband.table_index != 0)
        .unwrap_or(0);
    let spectral_subbands = quantized_subbands[..=last_coded]
        .iter()
        .map(|subband| subband.spectral_subband(coding_mode))
        .collect::<Result<Vec<_>>>()?;
    let payload_bits = quantized_subbands[..=last_coded]
        .iter()
        .map(|subband| subband.payload_bits)
        .sum();

    Ok(SpectrumEncoding {
        spectral_unit: SpectralUnit {
            coding_mode,
            subbands: spectral_subbands,
        },
        quantized_subbands,
        reconstructed: reconstructed.clone(),
        payload_bits,
        mse: mean_square(coefficients, &reconstructed),
    })
}

pub fn build_basic_sound_unit_from_encoding(encoding: &SpectrumEncoding) -> ChannelSoundUnit {
    let coded_qmf_bands = coded_qmf_bands_for_subband_count(encoding.spectral_unit.subbands.len());

    ChannelSoundUnit {
        coded_qmf_bands,
        gain_bands: vec![GainBand::default(); coded_qmf_bands as usize],
        tonal_mode_selector: TonalCodingModeSelector::AllVlc,
        tonal_components: Vec::new(),
        spectrum: encoding.spectral_unit.clone(),
    }
}

pub fn build_basic_sound_unit(
    coefficients: &[f32],
    coding_mode: CodingMode,
    options: SearchOptions,
) -> Result<ChannelSoundUnit> {
    let encoding = build_spectral_unit(coefficients, coding_mode, options)?;
    Ok(build_basic_sound_unit_from_encoding(&encoding))
}

#[derive(Debug, Clone)]
struct BudgetSolution {
    selected: Vec<QuantizedSubband>,
    used_bits: usize,
    mse: f32,
}

pub fn optimal_sf_index_for_peak(peak: f32, selector: u8) -> u8 {
    if peak <= 0.0 {
        return 0;
    }
    let imq = ATRAC3_INV_MAX_QUANT[selector as usize];
    if imq <= 0.0 {
        return 0;
    }
    let max_mantissa = match selector {
        1 => 1.0f32,
        2 => 2.0,
        3 => 3.0,
        4 => 4.0,
        5 => 7.0,
        6 => 15.0,
        7 => 31.0,
        _ => 1.0,
    };
    let needed_sf = peak / (max_mantissa * imq);
    let log_sf = needed_sf.max(1e-10).log2();
    // Ceil instead of round: guarantees the returned sfIndex accommodates
    // the peak without clipping. Round would land below the required value
    // about half the time and cause silent saturation at the codebook
    // boundary, showing up as worst-case errors much larger than the step.
    let index = (log_sf * 3.0 + 15.0).ceil() as i32;
    index.clamp(0, 63) as u8
}

fn collect_subband_candidates(
    coefficients: &[f32],
    coding_mode: CodingMode,
    options: SearchOptions,
) -> Result<Vec<QuantizedSubband>> {
    let mut best_by_bits = BTreeMap::<usize, QuantizedSubband>::new();
    let uncoded = QuantizedSubband::uncoded(coefficients);
    best_by_bits.insert(candidate_total_bits(&uncoded), uncoded);

    let peak = coefficients
        .iter()
        .map(|c| c.abs())
        .fold(0.0f32, f32::max);

    let sf_search_radius = 6u8;
    for selector in 1..=7u8 {
        let center = optimal_sf_index_for_peak(peak, selector);
        let lo = center.saturating_sub(sf_search_radius);
        let hi = (center + sf_search_radius).min(63);
        for scale_factor_index in lo..=hi {
            let candidate =
                quantize_subband(coefficients, selector, scale_factor_index, coding_mode)?;
            let total_bits = candidate_total_bits(&candidate);
            let replace = best_by_bits
                .get(&total_bits)
                .is_none_or(|current| candidate.mse + 1e-12 < current.mse);
            if replace {
                best_by_bits.insert(total_bits, candidate);
            }
        }
    }

    let mut frontier = Vec::with_capacity(best_by_bits.len());
    let mut best_mse = f32::INFINITY;
    for candidate in best_by_bits.into_values() {
        if candidate.mse + 1e-12 < best_mse {
            best_mse = candidate.mse;
            frontier.push(candidate);
        }
    }

    if frontier.len() > options.max_candidates_per_band {
        let len = frontier.len();
        let keep = options.max_candidates_per_band.max(2);
        let mut compact = Vec::with_capacity(keep);
        for index in 0..keep {
            let position = index * (len - 1) / (keep - 1);
            compact.push(frontier[position].clone());
        }
        compact.dedup_by(|left, right| {
            candidate_total_bits(left) == candidate_total_bits(right)
                && (left.mse - right.mse).abs() <= 1e-12
        });
        frontier = compact;
    }

    Ok(frontier)
}

fn solve_band_budget(
    candidates: &[Vec<QuantizedSubband>],
    target_bits: usize,
) -> Option<BudgetSolution> {
    let fixed_bits = fixed_sound_unit_bits(candidates.len());
    if fixed_bits > target_bits {
        return None;
    }

    let band_budget = target_bits - fixed_bits;
    let band_count = candidates.len();
    let state_count = band_budget + 1;
    let mut costs = vec![f32::INFINITY; (band_count + 1) * state_count];
    let mut parents = vec![usize::MAX; band_count * state_count];
    let mut choices = vec![usize::MAX; band_count * state_count];
    costs[0] = 0.0;

    for band_index in 0..band_count {
        let current = band_index * state_count;
        let next = (band_index + 1) * state_count;
        for used_bits in 0..=band_budget {
            let current_cost = costs[current + used_bits];
            if !current_cost.is_finite() {
                continue;
            }

            for (candidate_index, candidate) in candidates[band_index].iter().enumerate() {
                let next_bits = used_bits + candidate_total_bits(candidate);
                if next_bits > band_budget {
                    continue;
                }

                let next_cost = current_cost + candidate.mse;
                let slot = next + next_bits;
                if next_cost + 1e-12 < costs[slot] {
                    costs[slot] = next_cost;
                    parents[band_index * state_count + next_bits] = used_bits;
                    choices[band_index * state_count + next_bits] = candidate_index;
                }
            }
        }
    }

    let final_offset = band_count * state_count;
    let mut best_bits = None;
    let mut best_cost = f32::INFINITY;
    for used_bits in 0..=band_budget {
        let cost = costs[final_offset + used_bits];
        if cost + 1e-12 < best_cost
            || ((cost - best_cost).abs() <= 1e-12
                && best_bits.is_some_and(|current_bits| used_bits > current_bits))
        {
            best_cost = cost;
            best_bits = Some(used_bits);
        }
    }

    let mut used_bits = best_bits?;
    if !best_cost.is_finite() {
        return None;
    }

    let mut selected = Vec::with_capacity(band_count);
    for band_index in (0..band_count).rev() {
        let parent_slot = band_index * state_count + used_bits;
        let candidate_index = choices[parent_slot];
        if candidate_index == usize::MAX {
            return None;
        }
        selected.push(candidates[band_index][candidate_index].clone());
        used_bits = parents[parent_slot];
    }
    selected.reverse();

    let total_bits = fixed_bits + selected.iter().map(candidate_total_bits).sum::<usize>();
    Some(BudgetSolution {
        selected,
        used_bits: total_bits,
        mse: best_cost,
    })
}

fn finalize_budget_solution(
    coefficients: &[f32],
    coding_mode: CodingMode,
    mut solution: BudgetSolution,
) -> Result<SpectrumEncoding> {
    while solution.selected.len() > 1
        && solution
            .selected
            .last()
            .is_some_and(|candidate| candidate.table_index == 0)
    {
        solution.selected.pop();
    }

    let mut quantized_subbands = vec![QuantizedSubband::uncoded(&coefficients[0..8]); 32];
    let mut reconstructed = vec![0.0f32; coefficients.len()];
    let mut spectral_subbands = Vec::with_capacity(solution.selected.len());
    let mut payload_bits = 0usize;

    for (band, quantized) in solution.selected.iter().enumerate() {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        let reconstructed_band = quantized.dequantized(end - start)?;
        reconstructed[start..end].copy_from_slice(&reconstructed_band);
        quantized_subbands[band] = quantized.clone();
        spectral_subbands.push(quantized.spectral_subband(coding_mode)?);
        payload_bits += quantized.payload_bits;
    }

    for band in solution.selected.len()..32 {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        reconstructed[start..end].fill(0.0);
        quantized_subbands[band] = QuantizedSubband::uncoded(&coefficients[start..end]);
    }

    Ok(SpectrumEncoding {
        spectral_unit: SpectralUnit {
            coding_mode,
            subbands: spectral_subbands,
        },
        quantized_subbands,
        reconstructed: reconstructed.clone(),
        payload_bits,
        mse: mean_square(coefficients, &reconstructed),
    })
}

fn fixed_sound_unit_bits(subband_count: usize) -> usize {
    let coded_qmf_bands = coded_qmf_bands_for_subband_count(subband_count);
    6 + 2 + (coded_qmf_bands as usize * 3) + 5 + 5 + 1
}

fn candidate_total_bits(candidate: &QuantizedSubband) -> usize {
    3 + if candidate.table_index == 0 {
        0
    } else {
        6 + candidate.payload_bits
    }
}

fn build_spectral_unit_budgeted(
    coefficients: &[f32],
    coding_mode: CodingMode,
    _options: SearchOptions,
    target_bits: usize,
) -> Result<SpectrumEncoding> {
    ensure!(
        coefficients.len() == SAMPLES_PER_FRAME,
        "expected {} coefficients, got {}",
        SAMPLES_PER_FRAME,
        coefficients.len()
    );
    ensure!(target_bits > 0, "target_bits must be positive");

    // --- Phase 1: Compute per-band energy statistics ---
    let mut group_sf: Vec<Vec<u8>> = Vec::with_capacity(32);
    let mut band_peak_sf = [0i32; 32];
    let mut band_energy_sum = [0i32; 32];
    let mut num_active_bands = 0usize;

    for band in 0..32 {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        let groups = compute_group_sf_indices(&coefficients[start..end]);

        let mut peak: i32 = 0;
        let mut energy: i32 = 0;
        for &sf in &groups {
            let s = sf as i32;
            peak = peak.max(s);
            energy += s;
        }

        // Kill quiet bands (Sony energy threshold check)
        if energy < ENERGY_THRESHOLD[band] && peak < 3 {
            peak = 0;
            energy = 0;
        }

        band_peak_sf[band] = peak;
        band_energy_sum[band] = energy;
        if peak > 0 {
            num_active_bands = band + 1;
        }
        group_sf.push(groups);
    }

    if num_active_bands == 0 {
        num_active_bands = 1;
    }

    // --- Phase 2: Compute available budget ---
    let fixed_overhead = fixed_sound_unit_bits(num_active_bands);
    let available_bits = target_bits.saturating_sub(fixed_overhead) as i32;

    // --- Phase 3: Estimate cost at each tblIndex (1-7) for each band ---
    let mut tbl_indices = [0u8; 32];
    let mut sf_indices = [0i32; 32];

    // Pre-compute estimated costs: cost_at[band][tbl] = estimated bits * 10
    let mut cost_at = [[0i32; 8]; 32]; // [band][tblIndex]
    for band in 0..num_active_bands {
        if band_peak_sf[band] == 0 { continue; }
        sf_indices[band] = band_peak_sf[band];
        let width = ATRAC3_SUBBAND_TAB[band + 1] - ATRAC3_SUBBAND_TAB[band];
        for tbl in 1..=7u8 {
            cost_at[band][tbl as usize] = estimate_band_bit_cost(
                &group_sf[band], tbl, sf_indices[band], width,
            );
        }
    }

    // --- Phase 4: Mid-start budget allocation ---
    // Start at a moderate tblIndex proportional to peak sfIndex, then
    // demote if over budget, promote if under budget.
    let budget_10x = available_bits * 10;

    // Initial assignment: map peak sfIndex to a starting tblIndex,
    // with frequency-dependent boost for high bands to prevent dumpfness.
    let mut total_cost: i32 = 0;
    for band in 0..num_active_bands {
        if band_peak_sf[band] == 0 { continue; }
        // Base: peak/8 clamped to 1-7
        let mut initial = ((band_peak_sf[band] + 4) / 8).clamp(1, 7);
        // High-frequency boost: subbands above band 8 (>5.5kHz) get +1-2 tblIndex
        // to preserve brilliance/presence energy that low sfIndex would otherwise kill
        if band >= 16 && band_peak_sf[band] >= 2 {
            initial = (initial + 2).min(7);
        } else if band >= 8 && band_peak_sf[band] >= 2 {
            initial = (initial + 1).min(7);
        }
        tbl_indices[band] = initial as u8;
        total_cost += cost_at[band][initial as usize];
    }

    // Demote if over budget — but NEVER below tblIndex=1 for active bands.
    // Sony codes ALL subbands with at least tblIndex=1, preserving energy
    // across all frequencies. Skipping (tblIndex=0) causes the "dumpf" effect.
    while total_cost > budget_10x {
        let mut best_band = None;
        let mut best_savings: i32 = 0;
        let mut best_loss = f32::INFINITY;

        for band in 0..num_active_bands {
            let current_tbl = tbl_indices[band];
            if current_tbl <= 1 { continue; } // never demote below 1!
            let next_tbl = current_tbl - 1;
            let savings = cost_at[band][current_tbl as usize] - cost_at[band][next_tbl as usize];
            if savings <= 0 { continue; }
            let loss = band_peak_sf[band] as f32 / savings as f32;
            if best_band.is_none() || loss < best_loss {
                best_loss = loss;
                best_band = Some(band);
                best_savings = savings;
            }
        }

        match best_band {
            Some(band) => {
                total_cost -= best_savings;
                tbl_indices[band] -= 1;
            }
            None => break,
        }
    }

    // Promote if under budget (most important first)
    loop {
        let mut best_band = None;
        let mut best_efficiency = f32::NEG_INFINITY;
        let mut best_delta: i32 = 0;

        for band in 0..num_active_bands {
            let current_tbl = tbl_indices[band];
            if current_tbl == 0 || current_tbl >= 7 { continue; }
            let next_tbl = current_tbl + 1;
            let delta = cost_at[band][next_tbl as usize] - cost_at[band][current_tbl as usize];
            if delta <= 0 || total_cost + delta > budget_10x { continue; }
            let efficiency = band_peak_sf[band] as f32 / delta as f32;
            if efficiency > best_efficiency {
                best_efficiency = efficiency;
                best_band = Some(band);
                best_delta = delta;
            }
        }

        match best_band {
            Some(band) => {
                tbl_indices[band] += 1;
                total_cost += best_delta;
            }
            None => break,
        }
    }

    // --- Phase 5: Actual quantization with chosen parameters ---
    let mut quantized_subbands = Vec::with_capacity(num_active_bands);
    let mut reconstructed = vec![0.0f32; coefficients.len()];
    let mut spectral_subbands = Vec::with_capacity(num_active_bands);
    let mut payload_bits = 0usize;
    let mut used_bits = fixed_overhead;

    for band in 0..num_active_bands {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        let slice = &coefficients[start..end];

        if tbl_indices[band] == 0 || band_peak_sf[band] == 0 {
            let uncoded = QuantizedSubband::uncoded(slice);
            spectral_subbands.push(uncoded.spectral_subband(coding_mode)?);
            quantized_subbands.push(uncoded);
            used_bits += 3;
            continue;
        }

        // Use the chosen tblIndex and find the best sfIndex near the peak
        let selector = tbl_indices[band];
        let peak = slice.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        let sf_center = optimal_sf_index_for_peak(peak, selector);

        // sfIndex refinement: given the peak-fitting sfIndex (computed with
        // ceil, so it always accommodates the peak), search up to three
        // steps above it and pick the one that minimizes the worst-case
        // absolute reconstruction error. Going below sf_center would clip
        // the peak; going above trades precision for coarser grid which the
        // criterion will prefer only when the finer grid actually produces
        // a worse worst-case bin.
        let mut best: Option<QuantizedSubband> = None;
        let mut best_score = f32::INFINITY;
        for delta in 0i8..=3 {
            let sf_try = (sf_center as i8 + delta).clamp(0, 63) as u8;
            if let Ok(candidate) = quantize_subband(slice, selector, sf_try, coding_mode) {
                let bits = candidate_total_bits(&candidate);
                if used_bits + bits <= target_bits {
                    let score = candidate.max_abs_err;
                    if best.is_none() || score < best_score - 1e-12 {
                        best_score = score;
                        best = Some(candidate);
                    }
                }
            }
        }

        // Fallback: try lower codebooks if nothing fits
        if best.is_none() {
            for fallback_sel in (1..selector).rev() {
                let sf_fb = optimal_sf_index_for_peak(peak, fallback_sel);
                if let Ok(candidate) = quantize_subband(slice, fallback_sel, sf_fb, coding_mode) {
                    let bits = candidate_total_bits(&candidate);
                    if used_bits + bits <= target_bits {
                        best = Some(candidate);
                        break;
                    }
                }
            }
        }

        let quantized = best.unwrap_or_else(|| QuantizedSubband::uncoded(slice));
        used_bits += candidate_total_bits(&quantized);
        let reconstructed_band = quantized.dequantized(end - start)?;
        reconstructed[start..end].copy_from_slice(&reconstructed_band);
        payload_bits += quantized.payload_bits;
        spectral_subbands.push(quantized.spectral_subband(coding_mode)?);
        quantized_subbands.push(quantized);
    }

    // POST-PROMOTION: use real bit costs to fill remaining slot capacity.
    // Runs in multiple passes. Each pass walks the subbands in importance
    // order and tries to upgrade the table index by one, if the real bit
    // cost still fits in the remaining surplus. TZS and earlier upgrades
    // can free additional bits that become available to later passes, so
    // we iterate until no upgrade is found or surplus drops below a small
    // floor.
    let mut surplus = target_bits.saturating_sub(used_bits);
    for _pass in 0..4 {
        if surplus < 20 { break; }
        let mut order: Vec<usize> = (0..quantized_subbands.len().min(num_active_bands))
            .filter(|&b| {
                let t = quantized_subbands[b].table_index;
                t > 0 && t < 7
            })
            .collect();
        // Most important = highest peak in low-frequency bands first.
        order.sort_by(|&a, &b| {
            let pa = band_peak_sf[a] as f32;
            let pb = band_peak_sf[b] as f32;
            pb.partial_cmp(&pa).unwrap()
        });

        let mut upgrades_this_pass = 0usize;
        for &band in &order {
            if surplus < 20 { break; }
            let start = ATRAC3_SUBBAND_TAB[band];
            let end = ATRAC3_SUBBAND_TAB[band + 1];
            let slice = &coefficients[start..end];
            let cur_tbl = quantized_subbands[band].table_index;
            if cur_tbl >= 7 { continue; }
            let new_tbl = cur_tbl + 1;
            let peak = slice.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
            let sf_center = optimal_sf_index_for_peak(peak, new_tbl);
            let mut best_upgrade: Option<QuantizedSubband> = None;
            let mut best_score = quantized_subbands[band].mse;
            for delta in -2i8..=2 {
                let sf_try = (sf_center as i8 + delta).clamp(0, 63) as u8;
                if let Ok(cand) = quantize_subband(slice, new_tbl, sf_try, coding_mode) {
                    let extra = candidate_total_bits(&cand)
                        .saturating_sub(candidate_total_bits(&quantized_subbands[band]));
                    if extra <= surplus && cand.mse < best_score - 1e-12 {
                        best_score = cand.mse;
                        best_upgrade = Some(cand);
                    }
                }
            }
            if let Some(upgraded) = best_upgrade {
                let extra = candidate_total_bits(&upgraded)
                    .saturating_sub(candidate_total_bits(&quantized_subbands[band]));
                surplus = surplus.saturating_sub(extra);
                used_bits += extra;
                let recon = upgraded.dequantized(end - start)?;
                reconstructed[start..end].copy_from_slice(&recon);
                payload_bits = payload_bits + upgraded.payload_bits
                    - quantized_subbands[band].payload_bits;
                spectral_subbands[band] = upgraded.spectral_subband(coding_mode)?;
                quantized_subbands[band] = upgraded;
                upgrades_this_pass += 1;
            }
        }
        if upgrades_this_pass == 0 { break; }
    }

    // Trim trailing uncoded
    while spectral_subbands.len() > 1
        && spectral_subbands.last().is_some_and(|s| s.table_index == 0)
    {
        spectral_subbands.pop();
    }

    // Fill remaining bands
    for band in num_active_bands..32 {
        let start = ATRAC3_SUBBAND_TAB[band];
        let end = ATRAC3_SUBBAND_TAB[band + 1];
        reconstructed[start..end].fill(0.0);
        quantized_subbands.push(QuantizedSubband::uncoded(&coefficients[start..end]));
    }

    Ok(SpectrumEncoding {
        spectral_unit: SpectralUnit {
            coding_mode,
            subbands: spectral_subbands,
        },
        quantized_subbands,
        reconstructed: reconstructed.clone(),
        payload_bits,
        mse: mean_square(coefficients, &reconstructed),
    })
}

fn quantize_subband(
    coefficients: &[f32],
    selector: u8,
    scale_factor_index: u8,
    coding_mode: CodingMode,
) -> Result<QuantizedSubband> {
    let scale = scale_factor(scale_factor_index) * ATRAC3_INV_MAX_QUANT[selector as usize];
    ensure!(scale > 0.0, "selector {} has invalid scale", selector);

    let mut mantissas = match (coding_mode, selector) {
        (CodingMode::Clc, 1) => quantize_selector1_clc(coefficients, scale),
        (CodingMode::Vlc, 1) => quantize_selector1_vlc(coefficients, scale)?,
        (CodingMode::Clc, _) => quantize_signed_clc(coefficients, selector, scale),
        (CodingMode::Vlc, _) => quantize_vlc(coefficients, selector, scale)?,
    };

    // Trailing-Zero-Stripping: if all mantissas are even (and at least one
    // non-zero), halve them and add 3 to sfIndex. This is a mathematical
    // identity because scale_factor(sf+3) = 2 * scale_factor(sf), so the
    // dequantized values are bit-exact. The benefit is smaller absolute
    // mantissa values, which compress to shorter VLC codes.
    //
    // Only applied for selectors >= 2 because selector 1 uses a pair VLC
    // that depends on exact mantissa values, not just their scale.
    let mut final_sf = scale_factor_index;
    let mut final_scale = scale;
    if selector >= 2 && !mantissas.is_empty() {
        while final_sf <= 60 {
            let any_odd = mantissas.iter().any(|&m| m & 1 != 0);
            let any_nonzero = mantissas.iter().any(|&m| m != 0);
            if any_odd || !any_nonzero { break; }
            for m in &mut mantissas { *m /= 2; }
            final_sf += 3;
            final_scale = scale_factor(final_sf) * ATRAC3_INV_MAX_QUANT[selector as usize];
        }
    }

    let payload = encode_mantissas(selector, coding_mode, &mantissas)?;
    let reconstructed: Vec<f32> = mantissas
        .iter()
        .map(|&mantissa| mantissa as f32 * final_scale)
        .collect();

    // Track max-|error| per bin (worst-case selection criterion).
    let max_abs_err = coefficients.iter().zip(reconstructed.iter())
        .map(|(c, r)| (c - r).abs())
        .fold(0.0f32, f32::max);

    Ok(QuantizedSubband {
        table_index: selector,
        scale_factor_index: Some(final_sf),
        mantissas,
        payload_bits: payload.bit_len(),
        mse: mean_square(coefficients, &reconstructed),
        max_abs_err,
    })
}

fn quantize_selector1_clc(coefficients: &[f32], scale: f32) -> Vec<i8> {
    coefficients
        .iter()
        .map(|&coefficient| nearest_allowed(coefficient / scale, &ATRAC3_MANTISSA_CLC_TAB))
        .collect()
}

fn quantize_selector1_vlc(coefficients: &[f32], scale: f32) -> Result<Vec<i8>> {
    ensure!(
        coefficients.len() % 2 == 0,
        "selector 1 requires an even coefficient count"
    );

    let mut mantissas = Vec::with_capacity(coefficients.len());
    for pair in coefficients.chunks_exact(2) {
        let best_pair = ATRAC3_MANTISSA_VLC_PAIRS
            .iter()
            .copied()
            .min_by(|left, right| {
                pair_error(pair, *left, scale)
                    .partial_cmp(&pair_error(pair, *right, scale))
                    .unwrap()
            })
            .unwrap();
        mantissas.extend(best_pair);
    }
    Ok(mantissas)
}

fn quantize_signed_clc(coefficients: &[f32], selector: u8, scale: f32) -> Vec<i8> {
    let width = ATRAC3_CLC_LENGTH_TAB[selector as usize];
    let min_value = -(1i32 << (width - 1));
    let max_value = (1i32 << (width - 1)) - 1;

    coefficients
        .iter()
        .map(|&coefficient| {
            ((coefficient / scale).round() as i32)
                .clamp(min_value, max_value)
                .try_into()
                .unwrap()
        })
        .collect()
}

fn quantize_vlc(coefficients: &[f32], selector: u8, scale: f32) -> Result<Vec<i8>> {
    let allowed = huffman_codebooks()[selector as usize - 1]
        .iter()
        .map(|entry| entry.symbol)
        .collect::<Vec<_>>();
    Ok(coefficients
        .iter()
        .map(|&coefficient| nearest_allowed(coefficient / scale, &allowed))
        .collect())
}

fn nearest_allowed(value: f32, allowed: &[i8]) -> i8 {
    allowed
        .iter()
        .copied()
        .min_by(|left, right| {
            let left_error = (value - *left as f32).abs();
            let right_error = (value - *right as f32).abs();
            left_error.partial_cmp(&right_error).unwrap()
        })
        .unwrap()
}

fn pair_error(input: &[f32], candidate: [i8; 2], scale: f32) -> f32 {
    input
        .iter()
        .zip(candidate)
        .map(|(&sample, mantissa)| {
            let error = sample - mantissa as f32 * scale;
            error * error
        })
        .sum()
}

fn mean_square(reference: &[f32], candidate: &[f32]) -> f32 {
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(left, right)| {
            let error = left - right;
            error * error
        })
        .sum::<f32>()
        / reference.len().max(1) as f32
}

fn coded_qmf_bands_for_subband_count(subband_count: usize) -> u8 {
    let last_end = ATRAC3_SUBBAND_TAB[subband_count.clamp(1, 32)];
    ((((last_end.saturating_sub(1)) >> 8) + 1).clamp(1, 4)) as u8
}

fn twos_complement_bits(value: i32, bits: u8) -> u32 {
    if bits == 32 {
        value as u32
    } else {
        let mask = (1u32 << bits) - 1;
        (value as u32) & mask
    }
}

fn clc_symbol_index(value: i8) -> Result<u8> {
    ATRAC3_MANTISSA_CLC_TAB
        .iter()
        .position(|&candidate| candidate == value)
        .map(|index| index as u8)
        .ok_or_else(|| anyhow!("mantissa {} is not representable in selector-1 CLC", value))
}

fn vlc_pair_symbol(left: i8, right: i8) -> Result<i8> {
    ATRAC3_MANTISSA_VLC_PAIRS
        .iter()
        .position(|pair| pair[0] == left && pair[1] == right)
        .map(|index| index as i8)
        .ok_or_else(|| {
            anyhow!(
                "pair [{}, {}] is not representable in selector-1 VLC",
                left,
                right
            )
        })
}

fn find_huffman_entry(selector: u8, symbol: i8) -> Result<HuffmanEntry> {
    huffman_codebooks()[selector as usize - 1]
        .iter()
        .copied()
        .find(|entry| entry.symbol == symbol)
        .ok_or_else(|| {
            anyhow!(
                "symbol {} is not representable in selector {}",
                symbol,
                selector
            )
        })
}

fn huffman_codebooks() -> &'static [Vec<HuffmanEntry>] {
    static CODEBOOKS: OnceLock<Vec<Vec<HuffmanEntry>>> = OnceLock::new();
    CODEBOOKS.get_or_init(|| {
        let mut offset = 0usize;
        ATRAC3_HUFF_TAB_SIZES
            .iter()
            .map(|&size| {
                let next = offset + size as usize;
                let codebook = build_canonical_codebook(&ATRAC3_HUFF_TABS[offset..next]);
                offset = next;
                codebook
            })
            .collect()
    })
}

fn build_canonical_codebook(raw: &[(u8, u8)]) -> Vec<HuffmanEntry> {
    let max_bits = raw.iter().map(|entry| entry.1 as usize).max().unwrap_or(0);
    let mut counts = vec![0u32; max_bits + 1];
    for &(_, bits) in raw {
        counts[bits as usize] += 1;
    }

    let mut next_codes = vec![0u32; max_bits + 1];
    let mut code = 0u32;
    for bits in 1..=max_bits {
        code = (code + counts[bits - 1]) << 1;
        next_codes[bits] = code;
    }

    raw.iter()
        .map(|&(symbol, bits)| {
            let code = next_codes[bits as usize];
            next_codes[bits as usize] += 1;
            HuffmanEntry {
                symbol: symbol as i8 - 31,
                code,
                bits,
            }
        })
        .collect()
}

// ---------- Tonal extractor ----------

/// Result of tonal extraction: sound-unit-ready components plus a running
/// bit count that the spectral allocator must subtract from its budget.
pub struct TonalExtractionResult {
    pub tonal_mode_selector: super::sound_unit::TonalCodingModeSelector,
    pub tonal_components: Vec<super::sound_unit::TonalComponent>,
    pub tonal_bits: usize,
    pub coded_qmf_bands: u8,
}

/// Scan the residual for prominent peaks and encode them as tonal
/// components with qstep=7, CLC 6-bit, 4 values per entry. Subtracts each
/// committed tonal value from the residual in place so that the subsequent
/// spectral unit encodes only what is left over after removing the tones.
/// This avoids double-coding the same energy in both the tonal stream and
/// the spectral stream.
pub fn extract_tonal_components(
    residual: &mut [f32],
    budget_bits: usize,
    coded_qmf_bands: u8,
    _coding_mode: CodingMode,
    max_entries: usize,
) -> Result<TonalExtractionResult> {
    use super::sound_unit::*;

    let qmf_bands = coded_qmf_bands as usize;
    let total_cells = qmf_bands * 4;
    let spectral_end = (qmf_bands * 256).min(residual.len());
    let abs_threshold = 2.0f32;
    let qstep: u8 = 7;
    let clc_width = ATRAC3_CLC_LENGTH_TAB[qstep as usize];
    let imq = ATRAC3_INV_MAX_QUANT[qstep as usize];
    let max_mantissa = (1i32 << (clc_width - 1)) - 1;
    let min_mantissa = -(1i32 << (clc_width - 1));

    let entry_bits: usize = 12 + (clc_width as usize * 4);
    let new_band_bits: usize = 12;
    let base_bits: usize = 5 + 2 + qmf_bands + 3 + 3;
    let tonal_budget = budget_bits / 4;

    if tonal_budget < base_bits + entry_bits + new_band_bits {
        return Ok(TonalExtractionResult {
            tonal_mode_selector: TonalCodingModeSelector::AllVlc,
            tonal_components: Vec::new(),
            tonal_bits: 0,
            coded_qmf_bands,
        });
    }

    let mut cells: Vec<Vec<TonalEntry>> = vec![Vec::new(); total_cells];
    let mut band_active = vec![false; qmf_bands];
    let mut total_bits = base_bits;
    let mut total_entries: usize = 0;

    // First pass: collect all candidate chunks above threshold, each with
    // its peak magnitude. Candidates are always 4-aligned.
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    let mut pos = 0usize;
    while pos + 3 < spectral_end {
        let end = (pos + 4).min(spectral_end);
        let chunk = &residual[pos..end];
        let peak_val = chunk.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        if peak_val >= abs_threshold {
            candidates.push((pos, peak_val));
        }
        pos += 4;
    }

    // Sort by peak magnitude descending. Strongest peaks get extracted
    // first, regardless of frequency band, so the budget is spent on the
    // perceptually most prominent tones wherever they are in the spectrum.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (pos, _peak) in candidates {
        if total_entries >= max_entries {
            break;
        }
        let end = (pos + 4).min(spectral_end);
        let chunk = &residual[pos..end];
        let peak_val = chunk.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        if peak_val < abs_threshold {
            // A previous extraction may have reduced the peak below the
            // threshold. Skip, because what is left is not tonal any more.
            continue;
        }
        let sf_idx = optimal_sf_index_for_peak(peak_val, qstep);
        let sf_val = scale_factor(sf_idx);
        let scale = sf_val * imq;
        if scale < 1e-12 {
            continue;
        }

        let mut mantissas = [0i32; 4];
        let mut all_zero = true;
        for (i, m) in mantissas.iter_mut().enumerate() {
            if pos + i < spectral_end {
                *m = (residual[pos + i] / scale).round() as i32;
                *m = (*m).clamp(min_mantissa, max_mantissa);
                if *m != 0 {
                    all_zero = false;
                }
            }
        }
        if all_zero {
            continue;
        }

        let qmf_band = pos / 256;
        let cell_idx = pos / 64;
        if qmf_band >= qmf_bands || cell_idx >= total_cells {
            continue;
        }
        if cells[cell_idx].len() >= 7 {
            continue;
        }

        let band_cost = if band_active[qmf_band] { 0 } else { new_band_bits };
        if total_bits + band_cost + entry_bits > tonal_budget {
            continue;
        }

        // Subtract the quantized tonal contribution from the residual so
        // that the spectral allocator sees only what is left.
        for (i, &m) in mantissas.iter().enumerate() {
            if pos + i < spectral_end {
                residual[pos + i] -= m as f32 * scale;
            }
        }

        let mut payload = RawBitPayload::default();
        for &m in &mantissas {
            payload.push_bits((m as u32) & ((1u32 << clc_width) - 1), clc_width)?;
        }
        cells[cell_idx].push(TonalEntry {
            scale_factor_index: sf_idx,
            position: (pos % 64) as u8,
            payload,
        });

        if !band_active[qmf_band] {
            band_active[qmf_band] = true;
            total_bits += new_band_bits;
        }
        total_bits += entry_bits;
        total_entries += 1;
    }

    if total_entries == 0 {
        return Ok(TonalExtractionResult {
            tonal_mode_selector: TonalCodingModeSelector::AllVlc,
            tonal_components: Vec::new(),
            tonal_bits: 0,
            coded_qmf_bands,
        });
    }

    let tonal_cells: Vec<TonalCell> = cells
        .into_iter()
        .map(|entries| TonalCell { entries })
        .collect();
    let component = TonalComponent {
        band_flags: band_active,
        coded_values_minus_one: 3,
        quant_step_index: qstep,
        coding_mode: None,
        cells: tonal_cells,
    };
    Ok(TonalExtractionResult {
        tonal_mode_selector: TonalCodingModeSelector::AllClc,
        tonal_components: vec![component],
        tonal_bits: total_bits,
        coded_qmf_bands,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ATRAC3_CLC_LENGTH_TAB, ATRAC3_HUFF_TAB_SIZES, ATRAC3_INV_MAX_QUANT,
        ATRAC3_MANTISSA_CLC_TAB, ATRAC3_MANTISSA_VLC_TAB, QuantizedSubband, SearchOptions,
        build_basic_sound_unit, build_spectral_unit, choose_subband_encoding, clc_bit_width,
        encode_mantissas, find_huffman_entry, huff_table_size, inv_max_quant, scale_factor,
    };
    use crate::atrac3::{bitstream::BitWriter, sound_unit::CodingMode};

    fn payload_bytes(payload: &crate::atrac3::sound_unit::RawBitPayload) -> Vec<u8> {
        let mut writer = BitWriter::new();
        for chunk in &payload.chunks {
            writer.write_bits(chunk.value, chunk.bits).unwrap();
        }
        writer.byte_align_zero();
        writer.as_bytes().to_vec()
    }

    #[test]
    fn exposes_reference_clc_widths() {
        assert_eq!(ATRAC3_CLC_LENGTH_TAB, [0, 4, 3, 3, 4, 4, 5, 6]);
        assert_eq!(clc_bit_width(0), Some(0));
        assert_eq!(clc_bit_width(7), Some(6));
    }

    #[test]
    fn exposes_reference_mantissa_tables() {
        assert_eq!(ATRAC3_MANTISSA_CLC_TAB, [0, 1, -2, -1]);
        assert_eq!(
            ATRAC3_MANTISSA_VLC_TAB,
            [0, 0, 0, 1, 0, -1, 1, 0, -1, 0, 1, 1, 1, -1, -1, 1, -1, -1]
        );
    }

    #[test]
    fn exposes_reference_huffman_shapes() {
        assert_eq!(ATRAC3_HUFF_TAB_SIZES, [9, 5, 7, 9, 15, 31, 63]);
        assert_eq!(huff_table_size(1), Some(9));
        assert_eq!(huff_table_size(7), Some(63));
    }

    #[test]
    fn exposes_reference_inverse_quantizer() {
        assert_eq!(ATRAC3_INV_MAX_QUANT[0], 0.0);
        assert!((inv_max_quant(1).unwrap() - (1.0 / 1.5)).abs() < 1e-6);
        assert!((inv_max_quant(7).unwrap() - (1.0 / 31.5)).abs() < 1e-6);
    }

    #[test]
    fn encodes_selector1_clc_pairs() {
        let payload = encode_mantissas(1, CodingMode::Clc, &[1, -1, 0, -2]).unwrap();
        assert_eq!(payload.bit_len(), 8);
        assert_eq!(payload_bytes(&payload), vec![0x72]);
    }

    #[test]
    fn encodes_selector2_vlc_symbols() {
        let payload = encode_mantissas(2, CodingMode::Vlc, &[0, 1, -1, 2, -2]).unwrap();
        assert_eq!(payload.bit_len(), 13);
        assert_eq!(payload_bytes(&payload), vec![0x4b, 0xb8]);
    }

    #[test]
    fn selector1_pair_symbol_uses_five_bit_code() {
        let entry = find_huffman_entry(1, 5).unwrap();
        assert_eq!(entry.bits, 5);
        assert_eq!(entry.code, 0b11100);
    }

    #[test]
    fn silent_band_prefers_uncoded() {
        let band = vec![0.0f32; 8];
        let best =
            choose_subband_encoding(&band, CodingMode::Clc, SearchOptions::default()).unwrap();
        assert_eq!(best, QuantizedSubband::uncoded(&band));
    }

    #[test]
    fn builds_spectral_unit_and_trims_trailing_zero_subbands() {
        let mut spectrum = vec![0.0f32; 1024];
        spectrum[0] = scale_factor(15) * (1.0 / 2.5);

        let encoding = build_spectral_unit(
            &spectrum,
            CodingMode::Clc,
            SearchOptions {
                lambda: 0.0,
                ..SearchOptions::default()
            },
        )
        .unwrap();

        assert_eq!(encoding.spectral_unit.subbands.len(), 1);
        assert_ne!(encoding.spectral_unit.subbands[0].table_index, 0);
        assert!(encoding.payload_bits > 0);
    }

    #[test]
    fn builds_basic_sound_unit_from_quantized_spectrum() {
        let mut spectrum = vec![0.0f32; 1024];
        spectrum[600] = scale_factor(15) * (1.0 / 2.5);

        let unit = build_basic_sound_unit(
            &spectrum,
            CodingMode::Clc,
            SearchOptions {
                lambda: 0.0,
                ..SearchOptions::default()
            },
        )
        .unwrap();

        assert_eq!(unit.coded_qmf_bands, 3);
        assert_eq!(unit.gain_bands.len(), 3);
        assert!(unit.tonal_components.is_empty());
        assert!(unit.spectrum.subbands.len() >= 28);
    }
}
