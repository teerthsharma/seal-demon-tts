#!/bin/bash
# Master Quality Pipeline — RTX 4060 Optimized
# Trains Student TTS with quality-focused settings, then generates judged audiobook.
#
# Specs: RTX 4060 Laptop 8GB | i7-14700HX | batch_size=2 | accumulate=4

set -e

if command -v python &> /dev/null; then PYTHON=python
elif command -v python3 &> /dev/null; then PYTHON=python3
else echo "[ERROR] No python found"; exit 1; fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

LOGFILE="training_quality.log"
OUTPUT_DIR="checkpoints/student"
DATA_DIR="./data/student_pairs"
STUDENT_MAX_STEPS=15000
AUDIOBOOK_DIR="./audiobook/final_7hr"

# ============================================================
# STEP 1: Student Training (Quality Focused)
# ============================================================
echo "========================================" | tee -a $LOGFILE
echo "  DEMON TTS — QUALITY PIPELINE" | tee -a $LOGFILE
echo "  GPU: RTX 4060 Laptop 8GB" | tee -a $LOGFILE
echo "  batch_size=2 | accumulate=4 | fp16" | tee -a $LOGFILE
echo "  Start: $(date)" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# Find latest checkpoint for auto-resume
RESUME_ARG=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -t "$OUTPUT_DIR"/train-*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume_from_checkpoint $LATEST"
        echo "[QUALITY] Resuming from: $LATEST" | tee -a $LOGFILE
    else
        echo "[QUALITY] Starting from scratch (no checkpoint found)" | tee -a $LOGFILE
    fi
else
    echo "[QUALITY] Starting from scratch" | tee -a $LOGFILE
fi

# Ensure data exists
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.pt 2>/dev/null)" ]; then
    echo "[QUALITY] Generating student pairs..." | tee -a $LOGFILE
    $PYTHON convert_to_student_pairs.py 2>&1 | tee -a $LOGFILE
fi

# Ensure tokenizer exists
if [ ! -f "models/tokenizer.json" ]; then
    echo "[QUALITY] Training tokenizer..." | tee -a $LOGFILE
    $PYTHON train_tokenizer.py 2>&1 | tee -a $LOGFILE
fi

echo "[QUALITY] Starting training..." | tee -a $LOGFILE
set +e
$PYTHON training/train_student.py \
    --data_dir "$DATA_DIR" \
    --max_steps "$STUDENT_MAX_STEPS" \
    --batch_size 2 \
    --num_workers 0 \
    --output_dir "$OUTPUT_DIR" \
    --lr 2e-4 \
    --warmup_steps 4000 \
    $RESUME_ARG 2>&1 | tee -a $LOGFILE
TRAIN_EXIT=$?
set -e

if [ "$TRAIN_EXIT" -ne 0 ]; then
    if tail -30 "$LOGFILE" | grep -q "\[Export\] Saved to"; then
        echo "[WARN] Training exited with code $TRAIN_EXIT (Windows CUDA cleanup). Continuing..." | tee -a $LOGFILE
    else
        echo "[ERROR] Training failed with code $TRAIN_EXIT" | tee -a $LOGFILE
        echo "[ERROR] Check training_quality.log for details." | tee -a $LOGFILE
        exit $TRAIN_EXIT
    fi
fi

echo "[QUALITY] Training complete." | tee -a $LOGFILE

# ============================================================
# STEP 2: Judged Audiobook Generation
# ============================================================
echo "" | tee -a $LOGFILE
echo "[QUALITY] Starting judged audiobook generation..." | tee -a $LOGFILE

BOOK_COUNT=0
for f in book_parsed/*.json; do [ -f "$f" ] && BOOK_COUNT=$((BOOK_COUNT+1)); done

if [ "$BOOK_COUNT" -eq 0 ]; then
    echo "[ERROR] No parsed books found. Run convert_book.py first." | tee -a $LOGFILE
    exit 1
fi

set +e
$PYTHON generate_audiobook_judged.py \
    --book_dir ./book_parsed \
    --output_dir "$AUDIOBOOK_DIR" \
    --student_data_dir ./data/student_pairs_from_audiobook \
    --min_score 0.5 \
    --max_retries 2 2>&1 | tee -a $LOGFILE
AUDIO_EXIT=$?
set -e

if [ "$AUDIO_EXIT" -ne 0 ]; then
    if tail -30 "$LOGFILE" | grep -q "FULL BOOK:"; then
        echo "[WARN] Audiobook script exited $AUDIO_EXIT but output is valid." | tee -a $LOGFILE
    else
        echo "[ERROR] Audiobook generation failed with code $AUDIO_EXIT" | tee -a $LOGFILE
        exit $AUDIO_EXIT
    fi
fi

# ============================================================
# DONE
# ============================================================
echo "" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE
echo "  QUALITY PIPELINE COMPLETE" | tee -a $LOGFILE
echo "  End: $(date)" | tee -a $LOGFILE
echo "  Checkpoints: $OUTPUT_DIR/" | tee -a $LOGFILE
echo "  Audiobook: $AUDIOBOOK_DIR/" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# Prevent Windows CUDA cleanup crash from killing the pipeline
# Python scripts call os._exit(0) to bypass atexit handlers.
# Bash exits cleanly here.
exit 0
