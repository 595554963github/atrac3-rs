use anyhow::{Result, ensure};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::{
    prototype::{PrototypeEncodeResult, PrototypeEncoder, PrototypeOptions},
    sound_unit::{ChannelSoundUnit, CodingMode},
};
use crate::metrics::WavData;

pub const ATRAC3_WAVE_FORMAT_TAG: u16 = 0x0270;
pub const ATRAC3_FMT_CHUNK_SIZE: u32 = 32;
pub const ATRAC3_FACT_CHUNK_SIZE: u32 = 8;
pub const ATRAC3_WAV_EXTRADATA_SIZE: u16 = 14;
pub const ATRAC3_SAMPLES_PER_FRAME: u32 = 1024;
pub const ATRAC3_TARGET_SAMPLE_RATE: u32 = 44100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Atrac3Bitrate {
    Kbps66,
    Kbps105,
    Kbps132,
}

impl Atrac3Bitrate {
    pub fn kbps(self, channels: u16) -> u32 {
        match (channels, self) {
            (1, Self::Kbps66) => 33,
            (1, Self::Kbps105) => 52,
            (1, Self::Kbps132) => 66,
            (_, Self::Kbps66) => 66,
            (_, Self::Kbps105) => 105,
            (_, Self::Kbps132) => 132,
        }
    }

    pub fn block_align(self, channels: u16) -> u16 {
        let per_channel = match self {
            Self::Kbps66 => 96u16,
            Self::Kbps105 => 152u16,
            Self::Kbps132 => 192u16,
        };
        per_channel * channels
    }

    pub fn frame_factor(self) -> u16 {
        1
    }

    pub fn all() -> [Self; 3] {
        [Self::Kbps66, Self::Kbps105, Self::Kbps132]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Atrac3ContainerOptions {
    pub bitrate: Option<Atrac3Bitrate>,
}

impl Default for Atrac3ContainerOptions {
    fn default() -> Self {
        Self { bitrate: None }
    }
}

#[derive(Debug, Clone)]
pub struct Atrac3Container {
    pub bitrate: Atrac3Bitrate,
    pub block_align: u16,
    pub avg_bytes_per_sec: u32,
    pub bytes: Vec<u8>,
}

fn resample_wav(
    samples: &[f32],
    channels: u16,
    input_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>> {
    if input_rate == target_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        target_rate as f64 / input_rate as f64,
        2.0,
        params,
        samples.len() / channels as usize,
        channels as usize,
    )?;

    let channels_samples: Vec<Vec<f32>> = (0..channels)
        .map(|ch| {
            samples
                .iter()
                .skip(ch as usize)
                .step_by(channels as usize)
                .copied()
                .collect()
        })
        .collect();

    let resampled_channels = resampler.process(&channels_samples, None)?;
    
    let mut output = Vec::with_capacity(resampled_channels[0].len() * channels as usize);
    for i in 0..resampled_channels[0].len() {
        for ch in 0..channels as usize {
            output.push(resampled_channels[ch][i]);
        }
    }

    Ok(output)
}

pub fn wrap_prototype_in_riff_at3(
    encoded: &PrototypeEncodeResult,
    options: Atrac3ContainerOptions,
) -> Result<Atrac3Container> {
    let sample_rate = encoded.sample_rate;
    let channels = encoded.channel_count as u16;
    
    if sample_rate != ATRAC3_TARGET_SAMPLE_RATE {
        eprintln!("Resampling from {} Hz to {} Hz", sample_rate, ATRAC3_TARGET_SAMPLE_RATE);
        let resampled = resample_wav(
            &encoded.original_samples,
            channels,
            sample_rate,
            ATRAC3_TARGET_SAMPLE_RATE,
        )?;
        
        let wav = WavData {
            sample_rate: ATRAC3_TARGET_SAMPLE_RATE,
            channels,
            samples: resampled,
        };
        
        let resampled_encoded = PrototypeEncoder::encode_wav(&wav, encoded.options)?;
        return wrap_prototype_in_riff_at3(&resampled_encoded, options);
    }
    
    ensure!(
        encoded.channel_count >= 1 && encoded.channel_count <= 2,
        "prototype ATRAC3 container currently supports 1 or 2 channels, got {}",
        encoded.channel_count
    );
    ensure!(
        encoded.frame_count > 0,
        "cannot wrap an empty encode result"
    );

    let slot_size_for =
        |bitrate: Atrac3Bitrate| bitrate.block_align(channels) as usize / channels as usize;
    let max_channel_bytes = encoded
        .frames
        .iter()
        .flat_map(|frame| frame.channels.iter())
        .map(|channel| channel.bytes.len())
        .max()
        .unwrap_or(0);
    let bitrate = match options.bitrate {
        Some(bitrate) => {
            ensure!(
                !(channels == 2 && bitrate == Atrac3Bitrate::Kbps66),
                "stereo 66 kbps ATRAC3 requires joint stereo, which is not implemented yet"
            );
            ensure!(
                max_channel_bytes <= slot_size_for(bitrate),
                "encoded channel sound unit requires {} bytes, selected bitrate {} kbps only allows {} bytes per channel",
                max_channel_bytes,
                bitrate.kbps(channels),
                slot_size_for(bitrate)
            );
            bitrate
        }
        None => choose_smallest_fitting_bitrate(max_channel_bytes, channels)?,
    };

    let block_align = bitrate.block_align(channels);
    let channel_slot_size = block_align as usize / channels as usize;
    let avg_bytes_per_sec = ((block_align as u32 * ATRAC3_TARGET_SAMPLE_RATE)
        + (ATRAC3_SAMPLES_PER_FRAME / 2))
        / ATRAC3_SAMPLES_PER_FRAME;
    let sample_count = encoded.frame_count as u32 * ATRAC3_SAMPLES_PER_FRAME;
    let coding_mode = coding_mode_extradata(encoded);
    let data_size = encoded.frame_count as u32 * block_align as u32;

    let mut bytes = Vec::with_capacity(76 + data_size as usize);
    let riff_size =
        4 + (8 + ATRAC3_FMT_CHUNK_SIZE) + (8 + ATRAC3_FACT_CHUNK_SIZE) + (8 + data_size);

    bytes.extend_from_slice(b"RIFF");
    push_u32_le(&mut bytes, riff_size);
    bytes.extend_from_slice(b"WAVE");

    bytes.extend_from_slice(b"fmt ");
    push_u32_le(&mut bytes, ATRAC3_FMT_CHUNK_SIZE);
    push_u16_le(&mut bytes, ATRAC3_WAVE_FORMAT_TAG);
    push_u16_le(&mut bytes, channels);
    push_u32_le(&mut bytes, ATRAC3_TARGET_SAMPLE_RATE);
    push_u32_le(&mut bytes, avg_bytes_per_sec);
    push_u16_le(&mut bytes, block_align);
    push_u16_le(&mut bytes, 0);
    push_u16_le(&mut bytes, ATRAC3_WAV_EXTRADATA_SIZE);
    push_u16_le(&mut bytes, 1);
    push_u32_le(&mut bytes, 0x1000);
    push_u16_le(&mut bytes, coding_mode);
    push_u16_le(&mut bytes, coding_mode);
    push_u16_le(&mut bytes, bitrate.frame_factor());
    push_u16_le(&mut bytes, 0);

    bytes.extend_from_slice(b"fact");
    push_u32_le(&mut bytes, ATRAC3_FACT_CHUNK_SIZE);
    push_u32_le(&mut bytes, sample_count);
    push_u32_le(&mut bytes, ATRAC3_SAMPLES_PER_FRAME);

    bytes.extend_from_slice(b"data");
    push_u32_le(&mut bytes, data_size);

    for frame in &encoded.frames {
        ensure!(
            frame.channels.len() == encoded.channel_count,
            "frame contains {} channels, expected {}",
            frame.channels.len(),
            encoded.channel_count
        );

        for channel in &frame.channels {
            ensure!(
                channel.bytes.len() <= channel_slot_size,
                "channel sound unit {} exceeds per-channel slot {}",
                channel.bytes.len(),
                channel_slot_size
            );
            bytes.extend_from_slice(&channel.bytes);
            let padding = channel_slot_size - channel.bytes.len();
            if padding > 0 {
                bytes.resize(bytes.len() + padding, 0);
            }
        }
    }

    Ok(Atrac3Container {
        bitrate,
        block_align,
        avg_bytes_per_sec,
        bytes,
    })
}

fn choose_smallest_fitting_bitrate(
    max_channel_bytes: usize,
    channels: u16,
) -> Result<Atrac3Bitrate> {
    Atrac3Bitrate::all()
        .into_iter()
        .find(|bitrate| {
            !(channels == 2 && *bitrate == Atrac3Bitrate::Kbps66)
                && max_channel_bytes <= bitrate.block_align(channels) as usize / channels as usize
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "encoded channel sound unit requires {} bytes, exceeds supported ATRAC3 budgets",
                max_channel_bytes
            )
        })
}

fn coding_mode_extradata(encoded: &PrototypeEncodeResult) -> u16 {
    let has_joint_stereo = encoded
        .frames
        .iter()
        .flat_map(|frame| frame.channels.iter())
        .any(|channel| {
            matches!(channel.sound_unit.spectrum.coding_mode, CodingMode::Vlc)
                && channel.sound_unit.coded_qmf_bands > 1
                && encoded.channel_count == 2
                && has_joint_stereo_marker(&channel.sound_unit)
        });

    if has_joint_stereo { 1 } else { 0 }
}

fn has_joint_stereo_marker(_sound_unit: &ChannelSoundUnit) -> bool {
    false
}

fn push_u16_le(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32_le(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}