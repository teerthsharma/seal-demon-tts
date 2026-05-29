#!/usr/bin/env python3
"""
RunPod serverless handler for DemonTTS.
Deploy as a RunPod serverless endpoint.
"""
import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import runpod
import torch
import torchaudio

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from demo_tts import DemonTTS

# Global instance (cold start)
tts_engine = None


def init():
    """Initialize on cold start."""
    global tts_engine
    print("🚀 Cold start: Loading DemonTTS...")
    tts_engine = DemonTTS(device="cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Ready")


def handler(event):
    """RunPod serverless handler."""
    global tts_engine
    
    if tts_engine is None:
        init()
    
    job_input = event.get("input", {})
    text = job_input.get("text", "")
    voice_id = job_input.get("voice", "default")
    speaker_audio_b64 = job_input.get("speaker_audio", None)
    
    if not text:
        return {"error": "No text provided"}
    
    # Handle voice cloning
    speaker_emb = None
    if speaker_audio_b64:
        audio_bytes = base64.b64decode(speaker_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            speaker_emb = tts_engine.clone_voice(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
    elif voice_id in tts_engine.voices:
        speaker_emb = torch.tensor(tts_engine.voices[voice_id]).to(tts_engine.device)
    
    # Synthesize
    try:
        wav = tts_engine.synthesize(text, speaker_emb=speaker_emb)
    except Exception as e:
        return {"error": str(e)}
    
    # Encode
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_tensor, 24000, format="wav")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "audio": audio_b64,
        "format": "wav",
        "sample_rate": 24000,
        "duration_sec": len(wav) / 24000,
    }


# RunPod serverless entrypoint
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
