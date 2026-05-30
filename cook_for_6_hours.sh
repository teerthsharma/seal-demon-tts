#!/bin/bash
# DemonTTS 6-Hour Autonomous Training Pipeline
# Chapter 2 Master — Faraday + Aether training + Full book generation
# Run: bash cook_for_6_hours.sh
# Then walk away for 6 hours.

set -e

START_TIME=$(date +%s)
LOGFILE="training_6hr.log"
BOOK_JSON="book_parsed/Threshold's Pursuit_6b3bb078d03bc9c4.json"
MASTER_CHAPTER="2. Bound By Will"
OUTPUT_DIR="audiobook/final"
CHECKPOINT_DIR="checkpoints/autonomous"

echo "========================================" | tee -a $LOGFILE
echo "  DEMONTTS 6-HOUR AUTONOMOUS PIPELINE" | tee -a $LOGFILE
echo "  Master Chapter: $MASTER_CHAPTER" | tee -a $LOGFILE
echo "  Start: $(date)" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# --- STEP 0: Verify GPU ---
echo "" | tee -a $LOGFILE
echo "[0/6] GPU Check..." | tee -a $LOGFILE
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'CUDA: {torch.version.cuda}')
assert torch.cuda.is_available(), 'CUDA not available!'
" 2>&1 | tee -a $LOGFILE

# --- STEP 1: Restore old LSTM Aether architecture (checkpoint needs it) ---
echo "" | tee -a $LOGFILE
echo "[1/6] Restoring Aether LSTM architecture for checkpoint compatibility..." | tee -a $LOGFILE
git show 59f5bee^:aether/filter_net.py > aether/filter_net.py 2>/dev/null || echo "Using current filter_net.py"
git show 59f5bee^:aether/lattice_filter.py > aether/lattice_filter.py 2>/dev/null || echo "Using current lattice_filter.py"
echo "Aether architecture restored." | tee -a $LOGFILE

# --- STEP 2: Generate Chapter 2 Master Audio (clean pipeline) ---
echo "" | tee -a $LOGFILE
echo "[2/6] Generating Chapter 2 Master Audio..." | tee -a $LOGFILE
python -c "
import json
import soundfile as sf
from pathlib import Path
from pipeline_chapter2_master import DemonTTSMaster

Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)

tts = DemonTTSMaster(use_fp16=True)

with open('$BOOK_JSON', 'r', encoding='utf-8') as f:
    book = json.load(f)

chapter = book.get('$MASTER_CHAPTER')
if chapter:
    text = chapter.get('text', '')
    print(f'Chapter text: {len(text)} chars')
    wav = tts.synthesize(text)
    out_path = Path('$OUTPUT_DIR') / 'MASTER_chapter2.flac'
    sf.write(out_path, wav, 24000, format='FLAC')
    print(f'MASTER CHAPTER 2: {out_path} ({len(wav)/24000:.1f}s)')
else:
    print('ERROR: Chapter not found')
" 2>&1 | tee -a $LOGFILE

# --- STEP 3: Generate training pairs from Chapter 2 ---
echo "" | tee -a $LOGFILE
echo "[3/6] Generating Chapter 2 training pairs..." | tee -a $LOGFILE
python -c "
import json
import torch
import torchaudio
import random
import numpy as np
from pathlib import Path
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda')
processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
tts = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts', use_safetensors=True).to(device)
vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan', use_safetensors=True).to(device)
tts.eval()
vocoder.eval()

speaker_emb = torch.zeros(1, 512).to(device)

with open('$BOOK_JSON', 'r', encoding='utf-8') as f:
    book = json.load(f)

text = book.get('$MASTER_CHAPTER', {}).get('text', '')
sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if 20 < len(s.strip()) < 400]
print(f'Found {len(sentences)} sentences in Chapter 2')

# Generate 100 pairs
out_dir = Path('data/chapter2_pairs')
(out_dir / 'faraday_pairs').mkdir(parents=True, exist_ok=True)
(out_dir / 'aether_pairs').mkdir(parents=True, exist_ok=True)

for i in range(100):
    sent = random.choice(sentences)
    inputs = processor(text=sent, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        spectrogram = tts.generate_speech(inputs['input_ids'], speaker_emb)
        wav = vocoder(spectrogram)
    
    mel = spectrogram.T.unsqueeze(0).unsqueeze(0)  # [1, 1, 80, T]
    
    # Faraday pair: corrupt mel
    corrupted = mel.clone()
    noise = torch.randn_like(corrupted) * corrupted.std() * 0.15
    corrupted = corrupted + noise
    
    faraday_pair = {
        'student_mel': corrupted.cpu(),
        'gt_mel': mel.cpu(),
        'text_emb': torch.zeros(512).cpu(),
        'speaker_emb': speaker_emb.cpu().squeeze(0),
    }
    torch.save(faraday_pair, out_dir / 'faraday_pairs' / f'pair_{i:04d}.pt')
    
    # Aether pair: corrupt waveform
    wav_24k = torchaudio.transforms.Resample(16000, 24000)(wav)
    corrupted_wav = wav_24k + torch.randn_like(wav_24k) * wav_24k.std() * 0.1
    mel_aether = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)(corrupted_wav.unsqueeze(0))
    mel_aether = torch.log(mel_aether + 1e-6)
    T_mel = mel_aether.shape[2]
    
    aether_pair = {
        'input_waveform': corrupted_wav.unsqueeze(0).cpu(),
        'target_waveform': wav_24k.unsqueeze(0).cpu(),
        'mel': mel_aether.cpu(),
        'speaker_emb': torch.randn(192).cpu(),
        'f0': torch.zeros(1, T_mel).cpu(),
        'energy': mel_aether.mean(dim=1).cpu(),
    }
    torch.save(aether_pair, out_dir / 'aether_pairs' / f'pair_{i:04d}.pt')
    
    if (i+1) % 20 == 0:
        print(f'Generated {i+1}/100 pairs')

print('Chapter 2 pairs generated.')
" 2>&1 | tee -a $LOGFILE

# --- STEP 4: Train Faraday on Chapter 2 (resume from existing checkpoint) ---
echo "" | tee -a $LOGFILE
echo "[4/6] Training Faraday on Chapter 2... (this will take ~3 hours)" | tee -a $LOGFILE
echo "Training started at $(date)" | tee -a $LOGFILE

mkdir -p $CHECKPOINT_DIR/faraday_ch2

python -c "
import torch
import sys
from pathlib import Path

# Load trainer
sys.path.insert(0, 'training')
from train_faraday_supervised import *

# Override to use chapter2 data
class Args:
    data_dir = 'data/chapter2_pairs/faraday_pairs'
    output_dir = '$CHECKPOINT_DIR/faraday_ch2'
    epochs = 50
    batch_size = 1
    lr = 2e-4
    grad_accum = 8
    resume = 'checkpoints/faraday/epoch1_step700_emergency.pt'

args = Args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda')

model = FaradayDiffusion(text_dim=512, speaker_dim=512, cond_dim=512, base_channels=192).to(device)

# Load existing checkpoint
if args.resume and Path(args.resume).exists():
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Resumed from {args.resume}')
    print(f'Previous: epoch {ckpt.get(\"epoch\", \"?\")}, step {ckpt.get(\"step\", \"?\")}, loss {ckpt.get(\"loss\", \"?\")}')

model.unet.use_checkpoint = True
for m in model.unet.modules():
    if hasattr(m, 'checkpoint'):
        m.checkpoint = True

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

# Load dataset
dataset = FaradayPairDataset(args.data_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

print(f'Training on {len(dataset)} Chapter 2 pairs')
print(f'Target: {args.epochs} epochs')

best_loss = float('inf')
global_step = 0

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        student_mel = batch['student_mel'].to(device)
        gt_mel = batch['gt_mel'].to(device)
        text_emb = batch['text_emb'].to(device)
        speaker_emb = batch['speaker_emb'].to(device)
        
        with torch.cuda.amp.autocast():
            loss = model.supervised_training_loss(student_mel, gt_mel, text_emb, speaker_emb)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        
        epoch_loss += loss.item()
        
        if global_step % 10 == 0:
            print(f'Epoch {epoch+1}/{args.epochs} | Step {global_step} | Loss: {loss.item():.4f}', flush=True)
    
    avg_loss = epoch_loss / len(loader)
    print(f'=== Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} ===', flush=True)
    
    # Save checkpoint
    ckpt_path = Path(args.output_dir) / f'epoch{epoch}_loss{avg_loss:.4f}.pt'
    torch.save({
        'epoch': epoch,
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), Path(args.output_dir) / 'best.pt')
        print(f'*** New best: {best_loss:.4f} ***', flush=True)

print('Faraday training complete.')
" 2>&1 | tee -a $LOGFILE

# --- STEP 5: Train Aether on Chapter 2 ---
echo "" | tee -a $LOGFILE
echo "[5/6] Training Aether on Chapter 2..." | tee -a $LOGFILE

mkdir -p $CHECKPOINT_DIR/aether_ch2

python -c "
import torch
import sys
from pathlib import Path

sys.path.insert(0, 'training')
from train_aether_supervised import *

class Args:
    data_dir = 'data/chapter2_pairs/aether_pairs'
    output_dir = '$CHECKPOINT_DIR/aether_ch2'
    epochs = 50
    batch_size = 1
    lr = 1e-4
    grad_accum = 4
    resume = None

args = Args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda')
model = AetherFilter(lr=args.lr).to(device)

# Load existing if available
if args.resume and Path(args.resume).exists():
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    print(f'Resumed Aether from {args.resume}')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

dataset = WaveformPairDataset(args.data_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

print(f'Training Aether on {len(dataset)} pairs')

best_loss = float('inf')

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        input_wav = batch['input_waveform'].to(device)
        target_wav = batch['target_waveform'].to(device)
        mel = batch['mel'].to(device)
        speaker_emb = batch['speaker_emb'].to(device)
        f0 = batch['f0'].to(device)
        energy = batch['energy'].to(device)
        
        with torch.cuda.amp.autocast():
            out, loss = model(input_wav, mel, speaker_emb, f0, energy, target_wav)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f'Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}', flush=True)
    
    avg_loss = epoch_loss / len(loader)
    print(f'=== Aether Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ===', flush=True)
    
    ckpt_path = Path(args.output_dir) / f'epoch{epoch}_loss{avg_loss:.4f}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), Path(args.output_dir) / 'best.pt')
        print(f'*** Aether new best: {best_loss:.4f} ***', flush=True)

print('Aether training complete.')
" 2>&1 | tee -a $LOGFILE

# --- STEP 6: Copy trained models to models/ ---
echo "" | tee -a $LOGFILE
echo "[6/6] Copying trained checkpoints to models/..." | tee -a $LOGFILE

# Copy best Faraday
if [ -f "$CHECKPOINT_DIR/faraday_ch2/best.pt" ]; then
    cp $CHECKPOINT_DIR/faraday_ch2/best.pt models/faraday.pt
    echo "Faraday updated with Chapter 2 training." | tee -a $LOGFILE
fi

# Copy best Aether  
if [ -f "$CHECKPOINT_DIR/aether_ch2/best.pt" ]; then
    cp $CHECKPOINT_DIR/aether_ch2/best.pt models/aether.pt
    echo "Aether updated with Chapter 2 training." | tee -a $LOGFILE
fi

# --- STEP 7: Generate full book ---
echo "" | tee -a $LOGFILE
echo "[BONUS] Generating full audiobook with trained models..." | tee -a $LOGFILE

python -c "
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from pipeline_chapter2_master import DemonTTSMaster

Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)

tts = DemonTTSMaster(use_fp16=True)

with open('$BOOK_JSON', 'r', encoding='utf-8') as f:
    book = json.load(f)

pause = np.zeros(int(1.0 * 24000), dtype=np.float32)
parts = []
chapter_count = 0

for chapter_name, chapter_data in book.items():
    text = chapter_data.get('text', '')
    if not text.strip():
        continue
    
    chapter_count += 1
    print(f'[{chapter_count}] {chapter_name}...')
    
    try:
        wav = tts.synthesize(text)
        safe_name = ''.join(c if c.isalnum() else '_' for c in chapter_name)
        out_path = Path('$OUTPUT_DIR') / f'{safe_name}.flac'
        sf.write(out_path, wav, 24000, format='FLAC')
        print(f'  -> {out_path} ({len(wav)/24000:.1f}s)')
        parts.extend([wav, pause])
    except Exception as e:
        print(f'  -> FAILED: {e}')

if parts:
    parts.pop()  # Remove last pause
    combined = np.concatenate(parts)
    full_path = Path('$OUTPUT_DIR') / 'FULL_Thresholds_Pursuit.flac'
    sf.write(full_path, combined, 24000, format='FLAC')
    print(f'\nFULL BOOK: {full_path}')
    print(f'Total: {len(combined)/24000/60:.1f} minutes')
    print(f'Chapters: {chapter_count}')
" 2>&1 | tee -a $LOGFILE

# --- DONE ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))

echo "" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE
echo "  6-HOUR PIPELINE COMPLETE" | tee -a $LOGFILE
echo "  End: $(date)" | tee -a $LOGFILE
echo "  Duration: ${HOURS}h ${MINS}m" | tee -a $LOGFILE
echo "  Log: $LOGFILE" | tee -a $LOGFILE
echo "  Output: $OUTPUT_DIR/" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE
