use anyhow::{Result, ensure};
use std::sync::OnceLock;

use super::{
    SAMPLES_PER_FRAME,
    bitstream::BitWriter,
    gain::{GAIN_HISTORY_SLOTS, build_gain_curve, estimate_gain_band},
    mdct::{MDCT_COEFFS_PER_BAND, MDCT_INPUT_SAMPLES, Mdct256},
    qmf::{FourBandQmf, estimate_envelopes_from_interleaved},
    quant::{
        self, ATRAC3_SUBBAND_TAB, SearchOptions, SpectrumEncoding,
        build_basic_sound_unit_from_encoding, build_spectral_unit,
    },
    sound_unit::{ChannelSoundUnit, CodingMode, GainBand, RawBitPayload, SpectralSubband},
};
use crate::metrics::WavData;

const DEFAULT_QUANTIZER_COMPAT_GAIN: f32 = 7500.0;
const DEFAULT_APPLY_ODD_BAND_REVERSE: bool = true;
const DEFAULT_APPLY_GAIN_ESTIMATION: bool = false;
const DEFAULT_ANALYSIS_SAMPLE_OFFSET: isize = 69;
const DEFAULT_USE_REFERENCE_MDCT: bool = false;
const QMF_BAND_COUNT: usize = 4;

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

fn env_choice(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
}

fn quantizer_compat_gain() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("ATRAC3_QUANT_GAIN")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(DEFAULT_QUANTIZER_COMPAT_GAIN)
    })
}

fn apply_odd_band_reverse() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_flag("ATRAC3_ODD_REVERSE", DEFAULT_APPLY_ODD_BAND_REVERSE))
}

fn apply_gain_estimation() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| match env_choice("ATRAC3_GAIN").as_deref() {
        Some("0" | "false" | "no" | "off") => false,
        Some("1" | "true" | "yes" | "on") => true,
        _ => DEFAULT_APPLY_GAIN_ESTIMATION,
    })
}

fn analysis_sample_offset() -> isize {
    static VALUE: OnceLock<isize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("ATRAC3_ANALYSIS_SAMPLE_OFFSET")
            .ok()
            .and_then(|value| value.parse::<isize>().ok())
            .unwrap_or(DEFAULT_ANALYSIS_SAMPLE_OFFSET)
    })
}

fn swap_gain_curve_order() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_flag("ATRAC3_GAIN_CURVE_SWAP", false))
}

fn use_reference_mdct() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| env_flag("ATRAC3_REF_MDCT", DEFAULT_USE_REFERENCE_MDCT))
}

#[derive(Debug, Clone, Copy)]
enum MdctInputOrder {
    CurrentThenOverlap,
    OverlapThenCurrent,
}

impl MdctInputOrder {
    fn from_env() -> Self {
        match env_choice("ATRAC3_MDCT_INPUT_ORDER").as_deref() {
            Some("current-first" | "current_then_overlap") => Self::CurrentThenOverlap,
            Some("overlap-first" | "previous-first" | "overlap_then_current") => {
                Self::OverlapThenCurrent
            }
            _ => Self::OverlapThenCurrent,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MdctInputVariant {
    order: MdctInputOrder,
    reverse_first: bool,
    reverse_second: bool,
    negate_first: bool,
    negate_second: bool,
}

impl MdctInputVariant {
    fn from_env() -> Self {
        Self {
            order: MdctInputOrder::from_env(),
            reverse_first: env_flag("ATRAC3_MDCT_REVERSE_FIRST", false),
            reverse_second: env_flag("ATRAC3_MDCT_REVERSE_SECOND", false),
            negate_first: env_flag("ATRAC3_MDCT_NEGATE_FIRST", false),
            negate_second: env_flag("ATRAC3_MDCT_NEGATE_SECOND", false),
        }
    }

    fn build_input(
        &self,
        current: &[f32],
        overlap: &[f32],
    ) -> [f32; MDCT_INPUT_SAMPLES] {
        let mut current_block = [0.0f32; MDCT_COEFFS_PER_BAND];
        let mut overlap_block = [0.0f32; MDCT_COEFFS_PER_BAND];
        current_block.copy_from_slice(&current[..MDCT_COEFFS_PER_BAND]);
        overlap_block.copy_from_slice(&overlap[..MDCT_COEFFS_PER_BAND]);

        let (mut first, mut second) = match self.order {
            MdctInputOrder::CurrentThenOverlap => (current_block, overlap_block),
            MdctInputOrder::OverlapThenCurrent => (overlap_block, current_block),
        };

        if self.reverse_first {
            first.reverse();
        }
        if self.reverse_second {
            second.reverse();
        }
        if self.negate_first {
            for sample in &mut first {
                *sample = -*sample;
            }
        }
        if self.negate_second {
            for sample in &mut second {
                *sample = -*sample;
            }
        }

        let mut input = [0.0f32; MDCT_INPUT_SAMPLES];
        input[..MDCT_COEFFS_PER_BAND].copy_from_slice(&first);
        input[MDCT_COEFFS_PER_BAND..].copy_from_slice(&second);
        input
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrototypeOptions {
    pub coding_mode: CodingMode,
    pub lambda: f32,
    pub frame_limit: Option<usize>,
    pub start_frame: usize,
    pub flush_frames: usize,
    pub target_bits_per_channel: Option<usize>,
}

impl Default for PrototypeOptions {
    fn default() -> Self {
        Self {
            coding_mode: CodingMode::Clc,
            lambda: 0.0001,
            frame_limit: Some(1),
            start_frame: 0,
            flush_frames: 0,
            target_bits_per_channel: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrototypeFrameChannel {
    pub sound_unit: ChannelSoundUnit,
    pub spectrum: SpectrumEncoding,
    pub bit_len: usize,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PrototypeFrame {
    pub channels: Vec<PrototypeFrameChannel>,
    pub bit_len: usize,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PrototypeEncodeResult {
    pub sample_rate: u32,
    pub channel_count: usize,
    pub frame_count: usize,
    pub frames: Vec<PrototypeFrame>,
    pub bytes: Vec<u8>,
    pub original_samples: Vec<f32>,
    pub options: PrototypeOptions,
}

#[derive(Debug, Clone)]
pub struct GainInspectionBand {
    pub current_envelope: [f32; GAIN_HISTORY_SLOTS],
    pub previous_envelope: [f32; GAIN_HISTORY_SLOTS],
    pub gain_band: GainBand,
}

#[derive(Debug, Clone)]
pub struct GainInspectionChannel {
    pub bands: Vec<GainInspectionBand>,
}

#[derive(Debug, Clone)]
struct AnalyzedChannel {
    coefficients: Vec<f32>,
    gain_bands: Vec<GainBand>,
}

pub struct PrototypeEncoder {
    qmf: Vec<FourBandQmf>,
    mdct: Mdct256,
    overlap: Vec<[[f32; MDCT_COEFFS_PER_BAND]; 4]>,
    previous_gain_bands: Vec<Vec<GainBand>>,
    previous_envelopes: Vec<[[f32; GAIN_HISTORY_SLOTS]; QMF_BAND_COUNT]>,
    previous_peak_state: Vec<[f32; QMF_BAND_COUNT]>,
    pending_analysis: Vec<AnalyzedChannel>,
}

impl PrototypeEncoder {
    pub fn new(channel_count: usize) -> Self {
        Self {
            qmf: vec![FourBandQmf::default(); channel_count],
            mdct: Mdct256::default(),
            overlap: vec![[[0.0; MDCT_COEFFS_PER_BAND]; 4]; channel_count],
            previous_gain_bands: vec![vec![GainBand::default(); QMF_BAND_COUNT]; channel_count],
            previous_envelopes: vec![[[0.0; GAIN_HISTORY_SLOTS]; QMF_BAND_COUNT]; channel_count],
            previous_peak_state: vec![[0.0; QMF_BAND_COUNT]; channel_count],
            pending_analysis: vec![zero_analyzed_channel(); channel_count],
        }
    }

    pub fn encode_wav(wav: &WavData, options: PrototypeOptions) -> Result<PrototypeEncodeResult> {
        let channel_count = wav.channels as usize;
        ensure!(channel_count > 0, "WAV must contain at least one channel");

        let total_pcm_frames = wav.frames();
        let start_sample = options.start_frame * SAMPLES_PER_FRAME;
        ensure!(
            start_sample <= total_pcm_frames,
            "start_frame {} exceeds available PCM frames {}",
            options.start_frame,
            total_pcm_frames
        );
        let remaining_pcm_frames = total_pcm_frames.saturating_sub(start_sample);
        let input_frames = remaining_pcm_frames.div_ceil(SAMPLES_PER_FRAME);
        let available_frames = input_frames + options.flush_frames;
        ensure!(available_frames > 0, "no ATRAC3 frames available to encode");
        let frame_count = options
            .frame_limit
            .unwrap_or(available_frames)
            .min(available_frames);

        let channels = (0..channel_count)
            .map(|channel| wav.channel_samples(channel))
            .collect::<Result<Vec<_>>>()?;

        let mut encoder = PrototypeEncoder::new(channel_count);
        let mut all_analyses: Vec<Vec<AnalyzedChannel>> = Vec::with_capacity(frame_count);
        for frame_index in 0..frame_count {
            let start = start_sample + frame_index * SAMPLES_PER_FRAME;
            let frame_channels: Vec<Vec<f32>> = channels
                .iter()
                .map(|channel| slice_frame(channel, start))
                .collect();
            let analyses: Vec<AnalyzedChannel> = frame_channels
                .iter()
                .enumerate()
                .map(|(ch, samples)| encoder.analyze_channel_for_encoding(ch, samples))
                .collect::<Result<Vec<_>>>()?;
            encoder.pending_analysis = analyses.clone();
            all_analyses.push(analyses);
        }

        use rayon::prelude::*;
        let search_opts = SearchOptions {
            lambda: options.lambda,
            target_bits: options.target_bits_per_channel,
            max_candidates_per_band: 64,
            tonal_marked_subbands: [false; 32],
        };
        let frames: Vec<PrototypeFrame> = all_analyses
            .par_iter()
            .map(|analyses| {
                encoder.encode_analyzed_frame(analyses, options.coding_mode, search_opts)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut bytes = Vec::new();
        for f in &frames {
            bytes.extend_from_slice(&f.bytes);
        }

        Ok(PrototypeEncodeResult {
            sample_rate: wav.sample_rate,
            channel_count,
            frame_count,
            frames,
            bytes,
            original_samples: wav.samples.clone(),
            options,
        })
    }

    pub fn encode_frame(
        &mut self,
        channels: &[&[f32]],
        coding_mode: CodingMode,
        search: SearchOptions,
    ) -> Result<PrototypeFrame> {
        ensure!(
            channels.len() == self.qmf.len(),
            "channel count mismatch: got {}, encoder expects {}",
            channels.len(),
            self.qmf.len()
        );

        let current_analysis = channels
            .iter()
            .enumerate()
            .map(|(channel_index, samples)| {
                ensure!(
                    samples.len() == SAMPLES_PER_FRAME,
                    "prototype encoder expects {} samples per channel frame, got {}",
                    SAMPLES_PER_FRAME,
                    samples.len()
                );
                self.analyze_channel_for_encoding(channel_index, samples)
            })
            .collect::<Result<Vec<_>>>()?;

        self.pending_analysis = current_analysis.clone();
        self.encode_analyzed_frame(&current_analysis, coding_mode, search)
    }

    fn encode_analyzed_frame(
        &self,
        analysis_channels: &[AnalyzedChannel],
        coding_mode: CodingMode,
        search: SearchOptions,
    ) -> Result<PrototypeFrame> {
        ensure!(
            analysis_channels.len() == self.qmf.len(),
            "analysis channel count mismatch: got {}, encoder expects {}",
            analysis_channels.len(),
            self.qmf.len()
        );

        let mut frame_channels = Vec::with_capacity(analysis_channels.len());
        let mut frame_writer = BitWriter::new();

        for analysis in analysis_channels {
            let gain_payload_bits = analysis
                .gain_bands
                .iter()
                .map(|band| band.points.len() * 9)
                .sum::<usize>();
            let min_subband_count =
                minimum_subband_count_for_target_bits(search.target_bits.unwrap_or_default());
            let padded_skip_bits = min_subband_count.saturating_sub(1) * 3;
            let base_target = search.target_bits.unwrap_or(1536);
            let spectral_budget = base_target.saturating_sub(gain_payload_bits + padded_skip_bits);

            let coded_qmf_target: u8 = 3;
            let mut residual = analysis.coefficients.clone();
            let tonal_result = quant::extract_tonal_components(
                &mut residual,
                spectral_budget,
                coded_qmf_target,
                coding_mode,
                4,
            )?;

            let mut adjusted_search = search;
            adjusted_search.target_bits =
                Some(spectral_budget.saturating_sub(tonal_result.tonal_bits));
            adjusted_search.tonal_marked_subbands = tonal_result.tonal_subbands;

            let mut spectrum = build_spectral_unit(&residual, coding_mode, adjusted_search)?;
            pad_spectral_unit(&mut spectrum, min_subband_count);
            let mut sound_unit = build_basic_sound_unit_from_encoding(&spectrum);
            let coded_qmf_bands = sound_unit.coded_qmf_bands.min(coded_qmf_target) as usize;
            sound_unit.coded_qmf_bands = coded_qmf_bands as u8;
            sound_unit.gain_bands = analysis.gain_bands[..coded_qmf_bands].to_vec();

            sound_unit.tonal_mode_selector = tonal_result.tonal_mode_selector;
            sound_unit.tonal_components = tonal_result
                .tonal_components
                .into_iter()
                .map(|mut c| {
                    c.band_flags.resize(coded_qmf_bands, false);
                    c.cells.resize(
                        coded_qmf_bands * 4,
                        crate::atrac3::sound_unit::TonalCell::default(),
                    );
                    c
                })
                .collect();

            if let Ok(total) = sound_unit.bit_len() {
                if total > base_target {
                    sound_unit.tonal_mode_selector =
                        crate::atrac3::sound_unit::TonalCodingModeSelector::AllVlc;
                    sound_unit.tonal_components.clear();
                    adjusted_search.target_bits = Some(spectral_budget);
                    let s2 = build_spectral_unit(
                        &analysis.coefficients,
                        coding_mode,
                        adjusted_search,
                    )?;
                    sound_unit.spectrum = s2.spectral_unit;
                }
            }

            frame_channels.push(encode_channel(sound_unit, spectrum)?);
            frame_channels
                .last()
                .unwrap()
                .sound_unit
                .write_to(&mut frame_writer)?;
        }

        let frame_bits = frame_writer.bit_len();
        frame_writer.byte_align_zero();
        let frame_bytes = frame_writer.into_bytes();

        Ok(PrototypeFrame {
            channels: frame_channels,
            bit_len: frame_bits,
            bytes: frame_bytes,
        })
    }

    pub fn inspect_gain_frame(
        &mut self,
        channels: &[&[f32]],
    ) -> Result<Vec<GainInspectionChannel>> {
        ensure!(
            channels.len() == self.qmf.len(),
            "channel count mismatch: got {}, encoder expects {}",
            channels.len(),
            self.qmf.len()
        );

        let mut debug_channels = Vec::with_capacity(channels.len());
        for (channel_index, samples) in channels.iter().enumerate() {
            ensure!(
                samples.len() == SAMPLES_PER_FRAME,
                "prototype encoder expects {} samples per channel frame, got {}",
                SAMPLES_PER_FRAME,
                samples.len()
            );

            let frame = self.qmf[channel_index].split_frame_with_layout(samples)?;
            let envelopes = estimate_envelopes_from_interleaved(&frame.interleaved);
            let mut current_envelopes = [[0.0f32; GAIN_HISTORY_SLOTS]; QMF_BAND_COUNT];
            let mut current_gain_bands = vec![GainBand::default(); QMF_BAND_COUNT];
            let mut debug_bands = Vec::with_capacity(QMF_BAND_COUNT);

            for (band_index, _band_samples) in frame.bands.into_iter().enumerate() {
                let previous_envelope = self.previous_envelopes[channel_index][band_index];
                let current_envelope = envelopes[band_index];
                let history_peak_state = self.previous_peak_state[channel_index][band_index];
                let gain_band = if apply_gain_estimation() {
                    estimate_gain_band(
                        &current_envelope,
                        &previous_envelope,
                        band_index,
                        history_peak_state,
                    )
                } else {
                    GainBand::default()
                };

                current_envelopes[band_index] = current_envelope;
                current_gain_bands[band_index] = gain_band.clone();
                self.previous_peak_state[channel_index][band_index] =
                    previous_envelope.iter().copied().fold(0.0f32, f32::max);
                debug_bands.push(GainInspectionBand {
                    current_envelope,
                    previous_envelope,
                    gain_band,
                });
            }

            self.previous_envelopes[channel_index] = current_envelopes;
            self.previous_gain_bands[channel_index] = current_gain_bands;
            debug_channels.push(GainInspectionChannel { bands: debug_bands });
        }

        Ok(debug_channels)
    }

    pub fn analyze_frame_coefficients(&mut self, channels: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            channels.len() == self.qmf.len(),
            "channel count mismatch: got {}, encoder expects {}",
            channels.len(),
            self.qmf.len()
        );

        channels
            .iter()
            .enumerate()
            .map(|(channel_index, samples)| {
                ensure!(
                    samples.len() == SAMPLES_PER_FRAME,
                    "prototype encoder expects {} samples per channel frame, got {}",
                    SAMPLES_PER_FRAME,
                    samples.len()
                );
                self.analyze_channel_raw(channel_index, samples)
            })
            .collect()
    }

    fn analyze_channel_for_encoding(
        &mut self,
        channel_index: usize,
        samples: &[f32],
    ) -> Result<AnalyzedChannel> {
        self.analyze_channel(channel_index, samples, true)
    }

    fn analyze_channel_raw(&mut self, channel_index: usize, samples: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .analyze_channel(channel_index, samples, false)?
            .coefficients)
    }

    fn analyze_channel(
        &mut self,
        channel_index: usize,
        samples: &[f32],
        encode_gain: bool,
    ) -> Result<AnalyzedChannel> {
        let frame = self.qmf[channel_index].split_frame_with_layout(samples)?;
        let envelope_slots = estimate_envelopes_from_interleaved(&frame.interleaved);
        let mut coefficients = vec![0.0f32; SAMPLES_PER_FRAME];
        let gain_enabled = encode_gain && apply_gain_estimation();
        let mut gain_bands = vec![GainBand::default(); QMF_BAND_COUNT];
        let mut envelopes = [[0.0f32; GAIN_HISTORY_SLOTS]; QMF_BAND_COUNT];

        for (band_index, band_samples) in frame.bands.into_iter().enumerate() {
            let envelope = envelope_slots[band_index];
            envelopes[band_index] = envelope;
            let history_peak_state = self.previous_peak_state[channel_index][band_index];

            let gain_band = if gain_enabled {
                estimate_gain_band(
                    &envelope,
                    &self.previous_envelopes[channel_index][band_index],
                    band_index,
                    history_peak_state,
                )
            } else {
                GainBand::default()
            };
            gain_bands[band_index] = gain_band.clone();

            let analysis_samples = if gain_enabled {
                let curve = if swap_gain_curve_order() {
                    build_gain_curve(
                        &self.previous_gain_bands[channel_index][band_index],
                        &gain_band,
                    )?
                } else {
                    build_gain_curve(
                        &gain_band,
                        &self.previous_gain_bands[channel_index][band_index],
                    )?
                };
                compensate_band_samples(&band_samples, &curve.samples)
            } else {
                let mut out = [0.0f32; MDCT_COEFFS_PER_BAND];
                out.copy_from_slice(&band_samples[..MDCT_COEFFS_PER_BAND]);
                out
            };

            let mdct_input = MdctInputVariant::from_env().build_input(
                &analysis_samples,
                &self.overlap[channel_index][band_index],
            );
            self.overlap[channel_index][band_index].copy_from_slice(&analysis_samples);

            let mut band_coefficients = if use_reference_mdct() {
                self.mdct.forward_reference(&mdct_input)
            } else {
                let mut c = self.mdct.forward(&mdct_input);
                for coefficient in &mut c {
                    *coefficient *= quantizer_compat_gain();
                }
                c
            };
            if apply_odd_band_reverse() && (band_index & 1 == 1) {
                band_coefficients.reverse();
            }

            let start = band_index * MDCT_COEFFS_PER_BAND;
            let end = start + MDCT_COEFFS_PER_BAND;
            coefficients[start..end].copy_from_slice(&band_coefficients);
        }

        if gain_enabled {
            for band_index in 0..QMF_BAND_COUNT {
                self.previous_peak_state[channel_index][band_index] = self.previous_envelopes
                    [channel_index][band_index]
                    .iter()
                    .copied()
                    .fold(0.0f32, f32::max);
            }
            self.previous_envelopes[channel_index] = envelopes;
            self.previous_gain_bands[channel_index] = gain_bands.clone();
        }

        Ok(AnalyzedChannel {
            coefficients,
            gain_bands,
        })
    }
}

fn compensate_band_samples(
    band_samples: &[f32],
    curve: &[f32; MDCT_COEFFS_PER_BAND],
) -> [f32; MDCT_COEFFS_PER_BAND] {
    let mut out = [0.0f32; MDCT_COEFFS_PER_BAND];
    for index in 0..MDCT_COEFFS_PER_BAND {
        let gain = curve[index];
        out[index] = if gain.abs() > 1e-9 {
            band_samples[index] / gain
        } else {
            band_samples[index]
        };
    }
    out
}

fn zero_analyzed_channel() -> AnalyzedChannel {
    AnalyzedChannel {
        coefficients: vec![0.0; SAMPLES_PER_FRAME],
        gain_bands: vec![GainBand::default(); QMF_BAND_COUNT],
    }
}

fn minimum_subband_count_for_target_bits(target_bits: usize) -> usize {
    if target_bits >= 1_536 {
        ATRAC3_SUBBAND_TAB
            .iter()
            .position(|&end| end >= 768)
            .unwrap_or(0)
    } else {
        0
    }
}

fn pad_spectral_unit(encoding: &mut SpectrumEncoding, min_subband_count: usize) {
    while encoding.spectral_unit.subbands.len() < min_subband_count {
        encoding.spectral_unit.subbands.push(SpectralSubband {
            table_index: 0,
            scale_factor_index: None,
            payload: RawBitPayload::default(),
        });
    }
}

fn encode_channel(
    sound_unit: ChannelSoundUnit,
    spectrum: SpectrumEncoding,
) -> Result<PrototypeFrameChannel> {
    let mut writer = BitWriter::new();
    sound_unit.write_to(&mut writer)?;
    let bit_len = writer.bit_len();
    writer.byte_align_zero();

    Ok(PrototypeFrameChannel {
        sound_unit,
        spectrum,
        bit_len,
        bytes: writer.into_bytes(),
    })
}

fn slice_frame(channel: &[f32], start: usize) -> Vec<f32> {
    let mut frame = vec![0.0f32; SAMPLES_PER_FRAME];
    let shifted_start = start as isize + analysis_sample_offset();
    for (dst_index, sample) in frame.iter_mut().enumerate() {
        let src_index = shifted_start + dst_index as isize;
        if (0..channel.len() as isize).contains(&src_index) {
            *sample = channel[src_index as usize];
        }
    }
    frame
}