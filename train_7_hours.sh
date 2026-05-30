#!/bin/bash
# DemonTTS 7-Hour Autonomous Training Pipeline
# Idempotent — skips completed steps automatically.
# Run with:  bash train_7_hours.sh
# Force re-run everything:  NO_SKIP=1 bash train_7_hours.sh

set -e
set -o pipefail 2>/dev/null || true

# --- Detect Python ---
if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "[ERROR] Neither 'python' nor 'python3' found in PATH."
    exit 1
fi

# --- Config ---
NUM_PAIRS=2000
FARADAY_EPOCHS=30
AETHER_EPOCHS=25
LOGFILE="training_7hr.log"
OUTPUT_DIR="audiobook/final_7hr"

# Reduce CUDA memory fragmentation on 8GB cards
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --- Helper: count files matching a glob ---
count_files() {
    # $1 is a glob pattern; expand it, count actual files
    local count=0
    for f in $1; do
        if [ -f "$f" ]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# --- Idempotent skip logic ---
SKIP_DATA=0
SKIP_FARADAY=0
SKIP_AETHER=0
SKIP_STUDENT=0

if [ -z "$NO_SKIP" ] || [ "$NO_SKIP" != "1" ]; then
    FARADAY_COUNT=$(count_files "data/faraday_pairs/*.pt")
    if [ "$FARADAY_COUNT" -ge "$NUM_PAIRS" ]; then
        SKIP_DATA=1
        echo "[SKIP] Found $FARADAY_COUNT Faraday pairs (>= $NUM_PAIRS). Skipping data generation."
    fi

    if [ -f "checkpoints/faraday/best.pt" ]; then
        SKIP_FARADAY=1
        echo "[SKIP] Faraday checkpoint exists. Skipping Faraday training."
    fi

    if [ -f "checkpoints/aether/best.pt" ]; then
        SKIP_AETHER=1
        echo "[SKIP] Aether checkpoint exists. Skipping Aether training."
    fi

    # Student data auto-generated from existing pairs — never skip.
fi

START_TIME=$(date +%s)

echo "========================================" | tee -a $LOGFILE
echo "  DEMONTTS 7-HOUR AUTONOMOUS PIPELINE" | tee -a $LOGFILE
echo "  Voice: Human Male" | tee -a $LOGFILE
echo "  Start: $(date)" | tee -a $LOGFILE
echo "  Python: $($PYTHON --version 2>&1)" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# --- STEP 1: Data generation ---
if [ "$SKIP_DATA" -eq 1 ]; then
    echo "" | tee -a $LOGFILE
    echo "[1/4] Data generation SKIPPED (already have $FARADAY_COUNT pairs)." | tee -a $LOGFILE
else
    echo "" | tee -a $LOGFILE
    echo "[1/4] Generating $NUM_PAIRS training pairs..." | tee -a $LOGFILE
    $PYTHON generate_training_data.py --num_pairs $NUM_PAIRS --resume 2>&1 | tee -a $LOGFILE
fi

# --- STEP 2: Train Faraday & Aether ---
echo "" | tee -a $LOGFILE
if [ "$SKIP_FARADAY" -eq 1 ] && [ "$SKIP_AETHER" -eq 1 ]; then
    echo "[2/4] Faraday + Aether training SKIPPED (both checkpoints exist)." | tee -a $LOGFILE
elif [ "$SKIP_FARADAY" -eq 1 ]; then
    echo "[2/4] Training Aether only (~2 hours)..." | tee -a $LOGFILE
    set +e
    $PYTHON train_scheduler.py --faraday-epochs $FARADAY_EPOCHS --aether-epochs $AETHER_EPOCHS --auto-resume --skip-faraday 2>&1 | tee -a $LOGFILE
    TRAIN_EXIT=$?
    set -e
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        if tail -30 "$LOGFILE" | grep -q "ALL TRAINING COMPLETE"; then
            echo "[WARN] Scheduler exited with code $TRAIN_EXIT (likely Windows CUDA cleanup). Continuing..." | tee -a $LOGFILE
        else
            echo "[ERROR] Training failed with code $TRAIN_EXIT" | tee -a $LOGFILE
            exit $TRAIN_EXIT
        fi
    fi
elif [ "$SKIP_AETHER" -eq 1 ]; then
    echo "[2/4] Training Faraday only (~2 hours)..." | tee -a $LOGFILE
    set +e
    $PYTHON train_scheduler.py --faraday-epochs $FARADAY_EPOCHS --aether-epochs $AETHER_EPOCHS --auto-resume --skip-aether 2>&1 | tee -a $LOGFILE
    TRAIN_EXIT=$?
    set -e
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        if tail -30 "$LOGFILE" | grep -q "ALL TRAINING COMPLETE"; then
            echo "[WARN] Scheduler exited with code $TRAIN_EXIT (likely Windows CUDA cleanup). Continuing..." | tee -a $LOGFILE
        else
            echo "[ERROR] Training failed with code $TRAIN_EXIT" | tee -a $LOGFILE
            exit $TRAIN_EXIT
        fi
    fi
else
    echo "[2/4] Training Faraday + Aether (~4 hours)..." | tee -a $LOGFILE
    set +e
    $PYTHON train_scheduler.py --faraday-epochs $FARADAY_EPOCHS --aether-epochs $AETHER_EPOCHS --auto-resume 2>&1 | tee -a $LOGFILE
    TRAIN_EXIT=$?
    set -e
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        if tail -30 "$LOGFILE" | grep -q "ALL TRAINING COMPLETE"; then
            echo "[WARN] Scheduler exited with code $TRAIN_EXIT (likely Windows CUDA cleanup). Continuing..." | tee -a $LOGFILE
        else
            echo "[ERROR] Training failed with code $TRAIN_EXIT" | tee -a $LOGFILE
            exit $TRAIN_EXIT
        fi
    fi
fi

# --- STEP 3: Student Model ---
echo "" | tee -a $LOGFILE
echo "[3/4] Student distillation (~2 hours)..." | tee -a $LOGFILE

# Ensure tokenizer exists
if [ ! -f "models/tokenizer.json" ]; then
    echo "  [3/4a] Training BPE tokenizer on book text..." | tee -a $LOGFILE
    $PYTHON train_tokenizer.py 2>&1 | tee -a $LOGFILE
fi

# Ensure student pairs exist
STUDENT_COUNT=$(count_files "data/student_pairs/*.pt")
if [ "$STUDENT_COUNT" -eq 0 ]; then
    echo "  [3/4b] Converting existing pairs to student format..." | tee -a $LOGFILE
    $PYTHON convert_to_student_pairs.py 2>&1 | tee -a $LOGFILE
else
    echo "  [3/4b] Found $STUDENT_COUNT student pairs. Skipping conversion." | tee -a $LOGFILE
fi

echo "  [3/4c] Training Student Model..." | tee -a $LOGFILE
set +e
$PYTHON training/train_student.py --data_dir ./data/student_pairs --max_steps 15000 --batch_size 4 2>&1 | tee -a $LOGFILE
STUDENT_EXIT=$?
set -e
if [ "$STUDENT_EXIT" -ne 0 ]; then
    if tail -30 "$LOGFILE" | grep -q "\[Export\] Saved to"; then
        echo "[WARN] Student training exited with code $STUDENT_EXIT (likely Windows CUDA cleanup). Continuing..." | tee -a $LOGFILE
    else
        echo "[ERROR] Student training failed with code $STUDENT_EXIT" | tee -a $LOGFILE
        exit $STUDENT_EXIT
    fi
fi

# --- STEP 4: Audiobook ---
echo "" | tee -a $LOGFILE
echo "[4/4] Generating Full Audiobook..." | tee -a $LOGFILE

BOOK_COUNT=$(count_files "book_parsed/*.json")
if [ "$BOOK_COUNT" -eq 0 ]; then
    echo "[ERROR] No parsed books found in book_parsed/. Run convert_book.py first." | tee -a $LOGFILE
    exit 1
fi

$PYTHON -c "
import json, soundfile as sf, numpy as np, torch, os, sys
from pathlib import Path
from pipeline_chapter2_master import DemonTTSMaster

Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)

# Try fp16 first (fast, low VRAM), fallback to fp32 if dtype mismatch
for fp16 in [True, False]:
    try:
        tts = DemonTTSMaster(use_fp16=fp16)
        print(f'[Master] Using fp16={fp16}')
        break
    except Exception as e:
        print(f'[Master] fp16={fp16} init failed: {e}')
        if not fp16:
            raise

# Generate human male voice embedding
torch.manual_seed(1337)
base_male_vector = torch.randn(1, 512) * 0.1
base_male_vector[0, 10:50] += 0.5
base_male_vector[0, 100:150] -= 0.3
tts.speaker_emb = base_male_vector.to(tts.device)

# Load first parsed book
book_files = sorted(Path('book_parsed').glob('*.json'))
book_file = book_files[0]
with open(book_file, 'r', encoding='utf-8') as f:
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
    parts.pop()
    combined = np.concatenate(parts)
    full_path = Path('$OUTPUT_DIR') / 'FULL_Audiobook_Male_Voice.flac'
    sf.write(full_path, combined, 24000, format='FLAC')
    print(f'\nFULL BOOK: {full_path}')
    print(f'Total: {len(combined)/24000/60:.1f} minutes')

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
" 2>&1 | tee -a $LOGFILE

# --- DONE ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))

echo "" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE
echo "  PIPELINE COMPLETE" | tee -a $LOGFILE
echo "  End: $(date)" | tee -a $LOGFILE
echo "  Duration: ${HOURS}h ${MINS}m" | tee -a $LOGFILE
echo "  Output: $OUTPUT_DIR/" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE
