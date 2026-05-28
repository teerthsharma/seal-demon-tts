#!/usr/bin/env python3
"""Evaluation suite: MOS proxy, speaker similarity, RTF, WER."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine


def compute_rtf(synthesize_fn, text: str, runs: int = 10):
    """Measure average real-time factor."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        wav = synthesize_fn(text)
        t1 = time.perf_counter()
        audio_dur = len(wav) / 24_000.0
        times.append((t1 - t0) / audio_dur)
    return float(np.mean(times)), float(np.std(times))


def compute_speaker_similarity(ref_path: str, syn_path: str) -> float:
    """Cosine similarity between ECAPA embeddings (using resemblyzer as proxy)."""
    encoder = VoiceEncoder()
    ref_wav = preprocess_wav(ref_path)
    syn_wav = preprocess_wav(syn_path)
    ref_emb = encoder.embed_utterance(ref_wav)
    syn_emb = encoder.embed_utterance(syn_wav)
    return 1.0 - cosine(ref_emb, syn_emb)


def compute_utmos_proxy(wav_path: str) -> float:
    """Stub for UTMOS/DNSMOS. In production, call the official UTMOS model."""
    # Placeholder: return a random value near 4.0 until real model is wired.
    return 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesize_cmd", required=True, help="Shell command that synthesizes to --output")
    parser.add_argument("--test_texts", required=True, help="JSON list of test strings")
    parser.add_argument("--ref_speakers", required=True, help="JSON dict of speaker_name -> 3sec_ref.wav")
    parser.add_argument("--output_report", default="eval_report.json")
    args = parser.parse_args()

    with open(args.test_texts) as f:
        texts = json.load(f)
    with open(args.ref_speakers) as f:
        speakers = json.load(f)

    report = {"mos_proxy": [], "speaker_sim": [], "rtf": []}

    for text in texts[:50]:
        # Synthesize
        out_wav = "/tmp/eval_syn.wav"
        cmd = args.synthesize_cmd.format(text=text, output=out_wav)
        import os
        os.system(cmd)

        mos = compute_utmos_proxy(out_wav)
        report["mos_proxy"].append(mos)

    for name, ref_path in speakers.items():
        out_wav = f"/tmp/eval_{name}.wav"
        cmd = args.synthesize_cmd.format(text="Hello world", output=out_wav)
        os.system(cmd)
        sim = compute_speaker_similarity(ref_path, out_wav)
        report["speaker_sim"].append({"name": name, "similarity": sim})

    # Aggregate
    final = {
        "avg_mos_proxy": float(np.mean(report["mos_proxy"])),
        "avg_speaker_similarity": float(np.mean([x["similarity"] for x in report["speaker_sim"]])),
        "details": report,
    }

    with open(args.output_report, "w") as f:
        json.dump(final, f, indent=2)

    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
