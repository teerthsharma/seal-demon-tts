#!/bin/bash
# Standalone Student Training — run this if student failed in the main pipeline.
# Usage: bash train_student_only.sh

set -e

if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "[ERROR] Neither 'python' nor 'python3' found in PATH."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

LOGFILE="training_student.log"
DATA_DIR="${1:-./data/student_pairs}"
OUTPUT_DIR="${2:-checkpoints/student}"
MAX_STEPS="${3:-15000}"

echo "========================================" | tee -a $LOGFILE
echo "  STUDENT TRAINING (Standalone)" | tee -a $LOGFILE
echo "  Data: $DATA_DIR" | tee -a $LOGFILE
echo "  Output: $OUTPUT_DIR" | tee -a $LOGFILE
echo "  Max steps: $MAX_STEPS" | tee -a $LOGFILE
echo "  Start: $(date)" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# Ensure tokenizer exists
if [ ! -f "models/tokenizer.json" ]; then
    echo "[StudentOnly] Training BPE tokenizer..." | tee -a $LOGFILE
    $PYTHON train_tokenizer.py 2>&1 | tee -a $LOGFILE
fi

# Ensure student pairs exist
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.pt 2>/dev/null)" ]; then
    echo "[StudentOnly] Converting pairs to student format..." | tee -a $LOGFILE
    $PYTHON convert_to_student_pairs.py 2>&1 | tee -a $LOGFILE
fi

echo "[StudentOnly] Starting training..." | tee -a $LOGFILE
set +e
# Find latest checkpoint for auto-resume
RESUME_CKPT=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -t "$OUTPUT_DIR"/train-*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        RESUME_CKPT="--resume_from_checkpoint $LATEST"
        echo "[StudentOnly] Resuming from $LATEST" | tee -a $LOGFILE
    fi
fi

$PYTHON training/train_student.py \
    --data_dir "$DATA_DIR" \
    --max_steps "$MAX_STEPS" \
    --batch_size 2 \
    --num_workers 0 \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_CKPT 2>&1 | tee -a $LOGFILE
EXIT=$?
set -e

if [ "$EXIT" -ne 0 ]; then
    if tail -30 "$LOGFILE" | grep -q "\[Export\] Saved to"; then
        echo "[WARN] Exited $EXIT but export succeeded (Windows CUDA cleanup)." | tee -a $LOGFILE
    else
        echo "[ERROR] Training failed with code $EXIT" | tee -a $LOGFILE
        exit $EXIT
    fi
fi

echo "[StudentOnly] Done. Check $OUTPUT_DIR for checkpoints." | tee -a $LOGFILE
