#!/bin/bash
#SBATCH --job-name=chunked_decomp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Run your compression and evaluation script
python scripts/run_compression.py \
    --model_name "microsoft/DialoGPT-medium" \
    --output_dir "./results/experiment_1" \
    --batch_size 4 \
    --max_length 512 \
    --chunk_size 16 \
    --block_size 64 \
    --compression_ratios 0.25 0.5 0.75 \
    --benchmark_memory \
    --save_results

echo "Job completed!"
