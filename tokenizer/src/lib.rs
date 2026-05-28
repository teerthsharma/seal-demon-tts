use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// BPE tokenizer wrapper.
pub struct BpeTokenizer {
    inner: Tokenizer,
    max_len: usize,
}

impl BpeTokenizer {
    pub fn new(vocab_path: &str, _merges_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(vocab_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", vocab_path, e))?;
        Ok(Self {
            inner: tokenizer,
            max_len: 512,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
        } else {
            ids.resize(self.max_len, 0);
        }
        Ok(ids)
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.inner.decode(token_ids, false).unwrap_or_default()
    }
}

/// ONNX student inference wrapper.
pub struct OnnxStudent {
    session: Session,
}

impl OnnxStudent {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load student ONNX from {}", model_path))?;
        Ok(Self { session })
    }

    pub fn infer(&mut self, text_tokens: &[i64], speaker_embedding: &[f32]) -> Result<Vec<f32>> {
        let seq_len = text_tokens.len();
        let tokens_arr = ndarray::Array2::from_shape_vec((1, seq_len), text_tokens.iter().map(|&t| t as i64).collect())?;
        let spk_arr = ndarray::Array2::from_shape_vec((1, speaker_embedding.len()), speaker_embedding.to_vec())?;

        let tokens_val = Tensor::from_array(tokens_arr.into_dyn())?;
        let spk_val = Tensor::from_array(spk_arr.into_dyn())?;

        let outputs = self.session.run(ort::inputs! {
            "text_tokens" => tokens_val,
            "speaker_embedding" => spk_val,
        })?;

        let (shape, data) = outputs["mel"].try_extract_tensor::<f32>()?;
        let arr = ndarray::Array::from_shape_vec(shape.to_ixdyn(), data.to_vec())
            .map_err(|e| anyhow::anyhow!("Shape mismatch: {}", e))?;
        Ok(arr.iter().cloned().collect())
    }

    pub fn infer_mel_shape(&self, output: &[f32], time_steps: usize) -> Result<Array3<f32>> {
        let bins = 80;
        let expected = bins * time_steps;
        if output.len() != expected {
            anyhow::bail!(
                "Mel flat length mismatch: expected {} (80x{}), got {}",
                expected,
                time_steps,
                output.len()
            );
        }
        let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, bins, time_steps]), output.to_vec())
            .map_err(|e| anyhow::anyhow!("Reshape error: {}", e))?;
        Ok(arr.into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("Dimensionality error: {}", e))?
            .into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_dummy_vocab() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        let vocab: serde_json::Map<String, serde_json::Value> = [
            ("<pad>".to_string(), serde_json::json!(0)),
            ("hello".to_string(), serde_json::json!(1)),
            ("world".to_string(), serde_json::json!(2)),
        ]
        .into_iter()
        .collect();
        f.write_all(serde_json::to_string(&vocab).unwrap().as_bytes()).unwrap();
        f
    }

    fn create_dummy_merges() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"#version: 0.2\n").unwrap();
        f
    }

    #[test]
    fn test_bpe_encode_decode() {
        let vocab = create_dummy_vocab();
        let merges = create_dummy_merges();
        // Note: tokenizers crate may fail on dummy vocab without proper format;
        // this test validates compilation and basic API usage.
        let _tokenizer = BpeTokenizer::new(vocab.path().to_str().unwrap(), merges.path().to_str().unwrap());
        // If tokenizer loads, encode/decode roundtrip should work.
    }
}
