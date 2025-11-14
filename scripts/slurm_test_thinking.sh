#!/bin/bash
#SBATCH --job-name=qwen_thinking_test
#SBATCH --output=logs/qwen_thinking_%j.log
#SBATCH --error=logs/qwen_thinking_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# Test configurations for Qwen3-235B-A22B-Thinking model
# This script tests with 4 H100 GPUs first

echo "=========================================="
echo "Testing Qwen3-235B-A22B-Thinking-2507-FP8"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x H100"
echo "=========================================="

# Load modules
module load cuda/12.1
module load python/3.10

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nl2sql

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export HF_HOME=/scratch/$USER/huggingface_cache

# For distributed inference with transformers FP8
export CUDA_LAUNCH_BLOCKING=1

# Create directories
mkdir -p logs
mkdir -p results
mkdir -p $TRANSFORMERS_CACHE

# Print system information
echo "System Information:"
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# Test different prompts
PROMPTS=(
    "What are the key differences between transformer and LSTM architectures?"
    "Write a Python function to implement binary search on a sorted array."
    "Explain the concept of database normalization with examples."
)

# Run tests
for i in "${!PROMPTS[@]}"; do
    echo "Test $((i+1)): ${PROMPTS[$i]}"

    python scripts/test_qwen_hf.py \
        --model thinking \
        --num_gpus 4 \
        --max_tokens 1024 \
        --prompt "${PROMPTS[$i]}" \
        --output_file "results/thinking_test_${SLURM_JOB_ID}_prompt_$i.json"

    echo "Test $((i+1)) completed"
    echo "---"
done

echo "All tests completed for Qwen3-235B-A22B-Thinking"