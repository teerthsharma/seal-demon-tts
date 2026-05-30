#!/bin/bash
# DemonTTS Master Pipeline
# Must be launched via master.bat (admin mode + hygiene + pre-flight)
# Run standalone:  DO NOT. Use master.bat.

if [ -z "${MASTER_MODE:-}" ]; then
    echo "[ERROR] master.sh must be launched via master.bat"
    echo "  Run:  master.bat"
    exit 1
fi

set -euo pipefail 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

LOGFILE="training_7hr.log"

echo "========================================" | tee -a $LOGFILE
echo "  MASTER PIPELINE STARTING" | tee -a $LOGFILE
echo "  Admin mode: confirmed" | tee -a $LOGFILE
echo "  Start: $(date)" | tee -a $LOGFILE
echo "========================================" | tee -a $LOGFILE

# Delegate to the main 7-hour training script
bash train_7_hours.sh
