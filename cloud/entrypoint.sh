#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        DemonTTS Cloud GPU Node — Inkosei Engine         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo ""

# Check mounted volumes
if [ -d "/data" ]; then
    echo "📂 Data volume mounted at /data"
    mkdir -p /data/models /data/audiobook /data/checkpoints
fi

# Sync pretrained models from volume if available
if [ -f "/data/models/faraday.pt" ]; then
    echo "📥 Found Faraday checkpoint on volume"
    cp /data/models/*.pt ./models/ 2>/dev/null || true
fi

# Run based on MODE env var
case "${MODE:-inference}" in
    train)
        echo "🚀 Starting training orchestrator..."
        python3 train_all.py \
            --num_pairs "${NUM_PAIRS:-1000}" \
            --faraday_epochs "${FARADAY_EPOCHS:-50}" \
            --aether_epochs "${AETHER_EPOCHS:-50}" \
            --skip_data_gen "${SKIP_DATA_GEN:-false}"
        ;;
    inference)
        echo "🚀 Starting inference server..."
        python3 -m cloud.inference_server --host 0.0.0.0 --port 8000
        ;;
    batch)
        echo "🚀 Starting batch audiobook generation..."
        python3 cloud/batch_audiobook.py \
            --book_dir "${BOOK_DIR:-/data/book}" \
            --output_dir "${OUTPUT_DIR:-/data/audiobook}" \
            --voices "${VOICES_FILE:-./voices.json}"
        ;;
    data_gen)
        echo "🚀 Starting synthetic data generation..."
        python3 generate_training_data.py \
            --text_source "${TEXT_SOURCE:-/data/book}" \
            --output_dir "${OUTPUT_DIR:-/data}" \
            --num_pairs "${NUM_PAIRS:-1000}"
        ;;
    *)
        echo "Unknown MODE: ${MODE}"
        echo "Valid modes: train, inference, batch, data_gen"
        exit 1
        ;;
esac
