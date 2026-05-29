#!/usr/bin/env python3
"""
Modal.com deployment script for DemonTTS.
Run: modal deploy cloud/modal_deploy.py
"""
import modal

# Build the container image with CUDA support
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install(
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy<2.0.0",
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.7.0",
        "pypdf>=4.0.0",
        "pdfplumber>=0.11.0",
        "tqdm>=4.66.0",
        "huggingface-hub>=0.23.0",
    )
    .run_commands(
        "python3 -c \"from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan; "
        "SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts'); "
        "SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')\""
    )
)

# Create the app
app = modal.App("demon-tts", image=image)

# Volume for persistent model storage
models_volume = modal.Volume.from_name("demon-tts-models", create_if_missing=True)


@app.cls(
    gpu="T4",  # or "A10G", "A100", "H100"
    container_idle_timeout=300,
    volumes={"/models": models_volume},
    memory=16384,
)
class DemonTTSModal:
    def __init__(self):
        self.tts = None

    @modal.enter()
    def setup(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from demo_tts import DemonTTS
        self.tts = DemonTTS(device="cuda")
        print("✅ DemonTTS loaded on Modal GPU")

    @modal.method()
    def synthesize(self, text: str, voice_id: str = "default") -> bytes:
        import io
        import torch
        import torchaudio
        
        speaker_emb = None
        if voice_id in self.tts.voices:
            import torch as th
            speaker_emb = th.tensor(self.tts.voices[voice_id]).to(self.tts.device)
        
        wav = self.tts.synthesize(text, speaker_emb=speaker_emb)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        
        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor, 24000, format="wav")
        return buf.getvalue()

    @modal.method()
    def clone_voice(self, audio_bytes: bytes, name: str) -> dict:
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        
        try:
            emb = self.tts.clone_voice(tmp)
            self.tts.voices[name] = emb.cpu().tolist()
            return {"voice_id": name, "dims": len(emb)}
        finally:
            Path(tmp).unlink(missing_ok=True)

    @modal.web_endpoint(method="POST")
    def tts_endpoint(self, request: dict) -> dict:
        text = request.get("text", "")
        voice = request.get("voice", "default")
        
        audio_bytes = self.synthesize(text, voice)
        import base64
        return {
            "audio": base64.b64encode(audio_bytes).decode(),
            "format": "wav",
            "sample_rate": 24000,
        }


@app.local_entrypoint()
def main():
    tts = DemonTTSModal()
    audio = tts.synthesize.remote("Hello from Modal cloud compute!")
    print(f"Generated {len(audio)} bytes of audio")
