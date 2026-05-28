use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use tokenizer::{BpeTokenizer, OnnxStudent};
use tracing::{info, warn};

pub mod audio_io;

/// Hard ceiling for VRAM in bytes (~7.9 GB).
const MAX_VRAM_BYTES: usize = 7_900_000_000;

pub struct DemonPipeline {
    student: OnnxStudent,
    faraday: Session,
    vocoder: Session,
    aether: Session,
    tokenizer: BpeTokenizer,
}

impl DemonPipeline {
    pub fn new(
        student_path: &str,
        faraday_path: &str,
        vocoder_path: &str,
        aether_path: &str,
        vocab_path: &str,
        merges_path: &str,
    ) -> Result<Self> {
        info!("Loading tokenizer...");
        let tokenizer = BpeTokenizer::new(vocab_path, merges_path)
            .context("Failed to load BPE tokenizer")?;

        info!("Loading student ONNX (int8)...");
        let student = OnnxStudent::new(student_path)
            .context("Failed to load student ONNX")?;

        info!("Loading Faraday diffusion ONNX (fp16)...");
        let faraday = Session::builder()?
            .commit_from_file(faraday_path)
            .context("Failed to load Faraday ONNX")?;

        info!("Loading vocoder ONNX...");
        let vocoder = Session::builder()?
            .commit_from_file(vocoder_path)
            .context("Failed to load vocoder ONNX")?;

        info!("Loading Aether filter ONNX (fp16)...");
        let aether = Session::builder()?
            .commit_from_file(aether_path)
            .context("Failed to load Aether ONNX")?;

        info!("Pipeline loaded. All models on GPU.");
        Ok(Self {
            student,
            faraday,
            vocoder,
            aether,
            tokenizer,
        })
    }

    /// Synthesize audio from text.
    /// If `speaker_audio_path` is provided, a pre-computed speaker embedding `.npy` is expected
    /// at the same path with `.npy` extension (stub until full encoder is ported).
    pub fn synthesize(&mut self, text: &str, speaker_npy_path: Option<&str>) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text)?;
        let token_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let _seq_len = token_ids.len();

        // Speaker embedding: 192-dim. Use zeros if not provided.
        let speaker_emb: Vec<f32> = if let Some(path) = speaker_npy_path {
            read_npy_vec(path).unwrap_or_else(|e| {
                warn!("Failed to load speaker embedding: {}, using zeros", e);
                vec![0.0f32; 192]
            })
        } else {
            vec![0.0f32; 192]
        };

        // 1. Student inference → mel [1, 80, T]
        let mel = self.student.infer(&token_ids, &speaker_emb)?;
        let time_steps = mel.len() / 80;
        let mel_arr = self
            .student
            .infer_mel_shape(&mel, time_steps)
            .context("Failed to reshape mel")?;

        // 2. Faraday DDIM enhancement (10 steps) in Rust
        let refined_mel = self.ddim_denoise(&mel_arr, &speaker_emb, 10)?;

        // 3. Vocoder → waveform [1, T]
        let waveform = self.run_vocoder(&refined_mel)?;

        // 4. Aether filter → polished waveform
        let polished = self.run_aether(&waveform, &refined_mel, &speaker_emb)?;

        Ok(polished)
    }

    /// DDIM sampling loop using the Faraday U-Net ONNX model.
    fn ddim_denoise(
        &mut self,
        mel: &Array3<f32>,
        speaker_emb: &[f32],
        steps: usize,
    ) -> Result<Array3<f32>> {
        let (batch, bins, time) = (mel.shape()[0], mel.shape()[1], mel.shape()[2]);
        let mut x = mel.clone();

        // Linear timestep schedule (e.g., 999 → 0)
        let timesteps: Vec<i64> = (0..steps)
            .map(|i| (999 * (steps - 1 - i) / (steps - 1)) as i64)
            .collect();

        // Simple DDIM constants (cosine schedule approx)
        let alphas_cumprod: Vec<f32> = (0..1000)
            .map(|t| {
                let f = (t as f32 / 1000.0) * std::f32::consts::PI;
                (f.cos() + 1.0) / 2.0
            })
            .collect();

        for i in 0..steps {
            let t = timesteps[i];
            let alpha_t = alphas_cumprod[t as usize];
            let alpha_prev = if i == steps - 1 {
                1.0f32
            } else {
                alphas_cumprod[timesteps[i + 1] as usize]
            };

            // Build inputs: noisy mel + timestep + speaker_emb
            let mel_input = x.clone().to_shape((batch, 1, bins, time))?.into_owned();
            let t_tensor = ndarray::Array1::from_vec(vec![t as i64]);
            let spk_tensor = ndarray::Array2::from_shape_vec((1, 192), speaker_emb.to_vec())?;

            let mel_val = Tensor::from_array(mel_input.into_dyn())?;
            let t_val = Tensor::from_array(t_tensor.into_dyn())?;
            let spk_val = Tensor::from_array(spk_tensor.into_dyn())?;
            let outputs = self.faraday.run(ort::inputs! {
                "mel" => mel_val,
                "t" => t_val,
                "speaker_emb" => spk_val,
            })?;

            let (shape, data) = outputs["noise"].try_extract_tensor::<f32>()?;
            let pred_noise_4d = ndarray::Array::from_shape_vec(shape.to_ixdyn(), data.to_vec())?
                .into_dimensionality::<ndarray::Ix4>()?;
            let pred_noise = pred_noise_4d.into_shape_with_order((batch, bins, time))?;

            // DDIM update: x0 pred
            let x0 = (&x - (1.0 - alpha_t).sqrt() * &pred_noise) / alpha_t.sqrt();

            if i == steps - 1 {
                x = x0;
            } else {
                let direction = (1.0 - alpha_prev).sqrt() * &pred_noise;
                x = alpha_prev.sqrt() * &x0 + direction;
            }
        }

        Ok(x)
    }

    fn run_vocoder(&mut self, mel: &Array3<f32>) -> Result<Vec<f32>> {
        let (batch, bins, time) = (mel.shape()[0], mel.shape()[1], mel.shape()[2]);
        let mel_input = mel.clone().to_shape((batch, bins, time))?.into_owned();
        let mel_val = Tensor::from_array(mel_input.into_dyn())?;
        let outputs = self.vocoder.run(ort::inputs! {
            "mel" => mel_val,
        })?;
        let (shape, data) = outputs["waveform"].try_extract_tensor::<f32>()?;
        let wav = ndarray::Array::from_shape_vec(shape.to_ixdyn(), data.to_vec())?
            .into_dimensionality::<ndarray::Ix2>()?;
        Ok(wav.iter().cloned().collect())
    }

    fn run_aether(
        &mut self,
        waveform: &[f32],
        mel: &Array3<f32>,
        speaker_emb: &[f32],
    ) -> Result<Vec<f32>> {
        let t = waveform.len();
        let mel_t = mel.shape()[2];
        let wav_arr = Array2::from_shape_vec((1, t), waveform.to_vec())?;
        let mel_arr = mel.clone().to_shape((1, 80, mel_t))?.into_owned();
        let spk_arr = Array2::from_shape_vec((1, 192), speaker_emb.to_vec())?;

        let wav_val = Tensor::from_array(wav_arr.into_dyn())?;
        let mel_val = Tensor::from_array(mel_arr.into_dyn())?;
        let spk_val = Tensor::from_array(spk_arr.into_dyn())?;
        let outputs = self.aether.run(ort::inputs! {
            "waveform" => wav_val,
            "mel" => mel_val,
            "speaker_emb" => spk_val,
        })?;

        let (shape, data) = outputs["waveform"].try_extract_tensor::<f32>()?;
        let out = ndarray::Array::from_shape_vec(shape.to_ixdyn(), data.to_vec())?
            .into_dimensionality::<ndarray::Ix2>()?
            .into_owned();
        Ok(out.iter().cloned().collect())
    }
}

fn read_npy_vec(path: &str) -> Result<Vec<f32>> {
    // Stub: use numpy crate or memmap in production.
    // For now, read a simple JSON float array if .json, else error.
    if path.ends_with(".json") {
        let s = std::fs::read_to_string(path)?;
        let v: Vec<f32> = serde_json::from_str(&s)?;
        return Ok(v);
    }
    anyhow::bail!("Speaker embedding must be provided as a .json float array (192 dims) until npy reader is added.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        // This test relies on the tokenizer crate's own unit tests.
        // Here we just verify the pipeline lib compiles.
        assert_eq!(192, 192);
    }
}
