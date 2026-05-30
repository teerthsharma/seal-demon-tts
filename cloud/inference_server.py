#!/usr/bin/env python3
"""
FastAPI inference server for cloud-deployed DemonTTS.
Supports batch TTS, voice cloning, and health checks.
"""
import argparse
import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_tts import DemonTTS

app = FastAPI(title="DemonTTS Inference API", version="1.0.0")

# Global TTS instance (loaded once)
tts_engine = None


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speaker_audio: str | None = None  # base64-encoded WAV for cloning
    speed: float = 1.0
    format: str = "wav"  # wav, flac, mp3


class TTSResponse(BaseModel):
    audio: str  # base64-encoded
    format: str
    sample_rate: int
    duration_sec: float
    processing_time_sec: float


class HealthResponse(BaseModel):
    status: str
    gpu: str | None
    vram_gb: float | None
    models_loaded: bool


@app.on_event("startup")
async def startup():
    global tts_engine
    print("🚀 Loading DemonTTS engine...")
    tts_engine = DemonTTS()
    print(f"✅ Engine ready. GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_name = None
    vram_gb = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return HealthResponse(
        status="healthy",
        gpu=gpu_name,
        vram_gb=vram_gb,
        models_loaded=tts_engine is not None,
    )


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(req: TTSRequest):
    if tts_engine is None:
        raise HTTPException(500, "TTS engine not loaded")
    
    start = time.time()
    
    # Handle voice cloning if audio provided
    speaker_emb = None
    if req.speaker_audio:
        audio_bytes = base64.b64decode(req.speaker_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        speaker_emb = tts_engine.clone_voice(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    elif req.voice_id in tts_engine.voices:
        speaker_emb = np.array(tts_engine.voices[req.voice_id], dtype=np.float32)
    
    # Synthesize
    try:
        wav = tts_engine.synthesize(req.text, speaker_emb=speaker_emb)
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {str(e)}")
    
    # Speed adjustment if needed
    if req.speed != 1.0:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=int(24000 / req.speed)
        )(wav_tensor)
        wav = wav_tensor.squeeze().numpy()
    
    # Encode to requested format
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    buf = io.BytesIO()
    
    if req.format == "wav":
        torchaudio.save(buf, wav_tensor, 24000, format="wav")
    elif req.format == "flac":
        torchaudio.save(buf, wav_tensor, 24000, format="flac")
    else:
        raise HTTPException(400, f"Unsupported format: {req.format}")
    
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    
    proc_time = time.time() - start
    duration = len(wav) / 24000
    
    return TTSResponse(
        audio=audio_b64,
        format=req.format,
        sample_rate=24000,
        duration_sec=duration,
        processing_time_sec=proc_time,
    )


@app.post("/clone")
async def clone_voice(audio_b64: str, name: str):
    """Clone a voice from base64 audio and save it."""
    if tts_engine is None:
        raise HTTPException(500, "TTS engine not loaded")
    
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    
    try:
        emb = tts_engine.clone_voice(tmp_path)
        tts_engine.voices[name] = emb.tolist()
        tts_engine.save_voices()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    
    return {"voice_id": name, "dimensions": len(emb)}


@app.get("/voices")
async def list_voices():
    if tts_engine is None:
        raise HTTPException(500, "TTS engine not loaded")
    return {"voices": list(tts_engine.voices.keys())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
