#!/bin/bash

# Run comprehensive experiments across different configurations

echo "Starting ChunkedDecomp experiments..."

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results/experiments_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Experiment 1: Different compression ratios
echo "Running compression ratio experiments..."
python scripts/run_compression.py \
    --model_name "gpt2" \
    --output_dir "${RESULTS_DIR}/compression_ratios" \
    --chunk_size 16 \
    --block_size 64 \
    --compression_ratios 0.1 0.25 0.5 0.75 0.9

# Experiment 2: Different chunk sizes
echo "Running chunk size experiments..."
for chunk_size in 8 16 32; do
    python scripts/run_compression.py \
        --model_name "gpt2" \
        --output_dir "${RESULTS_DIR}/chunk_size_${chunk_size}" \
        --chunk_size ${chunk_size} \
        --block_size 64 \
        --compression_ratios 0.5
done

# Experiment 3: Memory benchmarks
echo "Running memory benchmarks..."
python scripts/benchmark_memory.py \
    --models "gpt2" "gpt2-medium" \
    --sequence_lengths 256 512 1024 \
    --output_dir "${RESULTS_DIR}/memory_benchmarks"

echo "Experiments completed! Results saved to ${RESULTS_DIR}"
