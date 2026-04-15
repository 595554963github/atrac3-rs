use anyhow::Result;
use atrac3_rs::atrac3::container::{
    Atrac3Bitrate, Atrac3ContainerOptions, wrap_prototype_in_riff_at3,
};
use atrac3_rs::atrac3::prototype::{PrototypeEncoder, PrototypeOptions};
use atrac3_rs::atrac3::sound_unit::CodingMode;
use atrac3_rs::metrics::read_wav;
use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about = "ATRAC3 encoder")]
struct Cli {
    #[arg(short = 'e')]
    encode: bool,

    input: PathBuf,

    #[arg(short, long, default_value_t = 132)]
    bitrate: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !cli.encode {
        eprintln!("Usage: at3cmp -e input.wav [--bitrate 132]");
        std::process::exit(1);
    }

    let output_path = cli.input.with_extension("at3");

    let wav = read_wav(&cli.input)?;
    
    let bitrate = match cli.bitrate {
        66 => Atrac3Bitrate::Kbps66,
        105 => Atrac3Bitrate::Kbps105,
        132 => Atrac3Bitrate::Kbps132,
        _ => {
            eprintln!("Unsupported bitrate {}, using 132 kbps", cli.bitrate);
            Atrac3Bitrate::Kbps132
        }
    };

    let target_bits_per_channel = (bitrate.block_align(wav.channels) as usize * 8) / wav.channels as usize;

    let encoded = PrototypeEncoder::encode_wav(
        &wav,
        PrototypeOptions {
            coding_mode: CodingMode::Clc,
            lambda: 0.0001,
            frame_limit: None,
            start_frame: 0,
            flush_frames: 0,
            target_bits_per_channel: Some(target_bits_per_channel),
        },
    )?;

    let at3 = wrap_prototype_in_riff_at3(
        &encoded,
        Atrac3ContainerOptions {
            bitrate: Some(bitrate),
        },
    )?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, &at3.bytes)?;

    println!("Encoded {} -> {}", cli.input.display(), output_path.display());
    println!("Original sample rate: {} Hz", wav.sample_rate);
    println!("Output sample rate: 44100 Hz");
    println!("Channels: {}", encoded.channel_count);
    println!("Frames: {}", encoded.frame_count);
    println!("Bitrate: {} kbps", bitrate.kbps(encoded.channel_count as u16));
    println!("File size: {} bytes", at3.bytes.len());

    Ok(())
}