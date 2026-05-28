# DemonTTS — The Inkosei Engine

> A decoupled FDFD solver disguised as an audiobook factory.

## What This Actually Is

DemonTTS is not just a TTS pipeline. It is a **multi-physics inference engine** where three neural networks collaborate to turn text into time-domain pressure waves (speech). The architecture is deliberately general-purpose:

- **Faraday** — A 2D Finite-Difference Frequency-Domain (FDFD) solver that operates on spectrogram-shaped fields. It can denoise electromagnetic fields, fluid pressure maps, or mel spectrograms. We happen to use it for audio.
- **Aether** — A differentiable IIR lattice filter bank that learns time-varying reflection coefficients. Originally designed for acoustic impedance matching; repurposed for waveform post-processing.
- **Student** — A 180M-parameter transformer backbone (distillation target).

## The Core Insight: Faraday as a Decoupled FDFD Solver

### FDFD Background
Finite-Difference Frequency-Domain solvers discretize a 2D spatial domain into a grid and solve the Helmholtz equation:

```
∇²E + k²ε(x,y)E = S(x,y)
```

where `E` is the field, `ε` is the permittivity, and `S` is the source. In matrix form:

```
A · x = b
```

`A` is a sparse system matrix encoding the Laplacian and material properties.

### Faraday's Mapping
Faraday's U-Net is a **learned preconditioner** for this class of PDEs:

| FDFD Concept | Faraday Implementation |
|--------------|------------------------|
| 2D spatial grid | Mel spectrogram `[B, 1, 80, T]` |
| Source term `b` | Noisy / corrupted input field |
| Field `E` | Clean target field |
| Material `ε(x,y)` | FiLM conditioning (text + speaker embedding) |
| System matrix `A` | Implicit in the U-Net convolution kernels |
| Residual prediction | U-Net output = `predicted_noise` or `predicted_residual` |

The **decoupling** refers to how the spatial operator (U-Net convolutions) is separated from the material properties (FiLM conditioning). You can change the text/speaker embedding (material properties) without retraining the spatial solver. This is exactly how physics-informed neural networks (PINNs) are structured.

### Multi-Purpose Usage

Because Faraday solves a **general field-to-field mapping**, it can be trained in multiple modes:

1. **Diffusion Mode** (generative): Predict noise in a noised field. Uses DDPM/DDIM sampling. Good for generative enhancement.
2. **Supervised Mode** (deterministic): Predict the residual `target - input` directly. No sampling. Good for denoising, deblurring, and restoration.
3. **Physical Mode** (PDE-constrained): Add a physics loss term (e.g., Helmholtz residual) to enforce that predictions satisfy Maxwell's equations.

### Audio-Specific Adaptation

For speech, the "field" is a log-mel spectrogram. The "material properties" are:
- **Text embedding**: Controls phonetic content (spatial source distribution)
- **Speaker embedding**: Controls timbre (frequency-dependent absorption/reflection)
- **Timestep embedding**: Controls noise level (analogous to frequency in FDFD)

The U-Net's skip connections act as **waveguide couplers** — they preserve high-frequency spatial detail that would otherwise be lost in the bottleneck.

## Architecture

```
Text ──→ Tacotron2 (pretrained) ──→ Mel [80, T]
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   Faraday Enhancer    │  ← FDFD solver on mel field
                              │  (U-Net + FiLM + DDIM)│
                              └───────────────────────┘
                                          │
                                          ▼
                                    Enhanced Mel
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   WaveRNN Vocoder     │  ← pretrained
                              │   (mel → waveform)    │
                              └───────────────────────┘
                                          │
                                          ▼
                                    Raw Waveform
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   Aether Filter Net   │  ← IIR lattice bank
                              │  (LSTM + 64 SOS filters)│
                              └───────────────────────┘
                                          │
                                          ▼
                                    Final Waveform @ 24kHz
```

### Parameter Budget

| Module | Params | Role | VRAM (fp16) |
|--------|--------|------|-------------|
| Tacotron2 | ~28M | Text → Mel | ~56 MB |
| WaveRNN | ~15M | Mel → Wave | ~30 MB |
| Faraday U-Net | ~22M | Mel enhancement (FDFD) | ~44 MB |
| Aether Filter | ~0.8M | Waveform polish | ~1.6 MB |
| **Total** | **~66M** | | **~132 MB** |

Fits in 8GB VRAM with room for 4K-page audiobooks.

## Training

### Prerequisites

```bash
pip install torch torchaudio numpy soundfile pygame customtkinter pypdf pdfplumber tokenizers transformers
```

### Phase 1: Generate Synthetic Training Data

We use pretrained Tacotron2 + WaveRNN as a "teacher" to generate clean mel/waveform pairs, then add synthetic corruption to create input-target pairs.

```bash
python generate_training_data.py \
  --text_source ./book/ \
  --output_dir ./data \
  --num_pairs 5000
```

**Corruption strategies:**
- **Faraday**: Gaussian noise + spectral masking + time masking + mild blur on mel
- **Aether**: Gaussian noise + codec compression + mild clipping on waveform

No human recordings required.

### Phase 2: Train Faraday (Supervised Mode)

```bash
python training/train_faraday.py \
  --data_dir ./data/faraday_pairs \
  --output_dir checkpoints/faraday \
  --batch_size 16 \
  --max_steps 30000
```

In supervised mode, Faraday predicts the residual `enhanced_mel - corrupted_mel` directly. Loss is L1 on mel values. The diffusion scheduler is bypassed (t=0 always).

**Time**: ~6-10 hours on RTX 4060

### Phase 3: Train Aether

```bash
python training/train_aether.py \
  --data_dir ./data/aether_pairs \
  --output_dir checkpoints/aether \
  --batch_size 8 \
  --max_steps 15000
```

Loss: Multi-resolution STFT + perceptual (mel L1).

**Time**: ~3-5 hours on RTX 4060

### Phase 4: Run the Audiobook Generator

```bash
python gui.py
```

Or batch convert:
```bash
python convert_book.py
```

## Faraday's General-Purpose API

Because Faraday is fundamentally an FDFD solver, you can use it for any 2D field enhancement task:

```python
from faraday.model import FaradayDiffusion

solver = FaradayDiffusion(
    text_dim=512,      # conditioning dimension
    speaker_dim=256,   # secondary conditioning
    cond_dim=128,      # fused conditioning width
    base_channels=64,  # spatial resolution
)

# Mode 1: Diffusion (generative)
enhanced = solver.enhance(corrupted_field, steps=10)

# Mode 2: Supervised (deterministic)
# Modify model.forward to predict residual directly
# Train with L1(target, input + residual)
```

The only requirement is that your input field has shape `[B, 1, H, W]` where `H` and `W` are spatial dimensions. For audio, `H=80` (mel bins) and `W=T` (time frames). For EM simulations, `H` and `W` would be the Yee-cell grid dimensions.

## Aether's Differentiable IIR Bank

Aether uses a parallel bank of 64 second-order IIR filters (SOS sections) whose coefficients are predicted by an LSTM from mel + speaker + f0 + energy features.

```python
from aether.model import AetherFilter

filter_net = AetherFilter()
refined_waveform = filter_net(
    waveform=wav,           # [B, 1, T]
    mel=mel,                # [B, 80, T_mel]
    speaker_emb=spk,        # [B, 192]
    f0=f0,                  # [B, 1, T_mel]
    energy=energy,          # [B, 1, T_mel]
)
```

The lattice structure guarantees stability: all poles are inside the unit circle because reflection coefficients are bounded by the tanh activation in the LSTM output.

## The Rust Pipeline (Optional)

For production inference, the Python models can be exported to ONNX and run via the Rust pipeline:

```bash
cd pipeline
cargo run --release --bin demon-tts
```

The Rust crate uses `ort` (ONNX Runtime) with CUDA EP for zero-overhead inference.

## Voice Cloning

Zero-shot speaker cloning via ECAPA-TDNN encoder:

```python
from demo_tts import DemonTTS

tts = DemonTTS()
embedding = tts.clone_voice("speaker_3sec_sample.wav")
tts.voices["My Clone"] = embedding
tts.save_voices()
```

Only 3 seconds of audio needed. The encoder extracts a 192-dimensional speaker embedding from mel-spectrogram statistics.

## Folder Structure

```
./book/              # Input PDFs
./book_parsed/       # Cached JSON chapters
./audiobook/         # Output FLAC + combined audiobook
./data/              # Synthetic training pairs
./models/            # Checkpoints (.pt) + tokenizer + voices
./faraday/           # FDFD solver core
./aether/            # IIR lattice filter bank
./neural/            # Student + SpeakerEncoder + HiFi-GAN
./pipeline/          # Rust ONNX inference engine
./training/          # Lightning training scripts
./gui.py             # CustomTkinter audiobook factory
```

## License

MIT. Do whatever. Build a multiverse. Don't blame us if Morty presses the wrong button.

---

*Wubba lubba dub dub.*
