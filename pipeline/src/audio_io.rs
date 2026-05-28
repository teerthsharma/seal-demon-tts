use anyhow::Result;
use hound::{WavSpec, WavWriter};
use rubato::{FftFixedInOut, Resampler};

/// Write f32 samples to a mono WAV file at the given sample rate.
pub fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(int_sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Resample f32 mono audio using FFT-based resampling.
pub fn resample(input: &[f32], from_sr: usize, to_sr: usize) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(input.to_vec());
    }
    let mut resampler = FftFixedInOut::<f32>::new(from_sr, to_sr, 1024, 1)?;
    let waves_in = vec![input.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out.into_iter().next().unwrap_or_default())
}
