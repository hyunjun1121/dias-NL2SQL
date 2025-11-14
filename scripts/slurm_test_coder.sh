#!/bin/bash
#SBATCH --job-name=qwen_coder_test
#SBATCH --output=logs/qwen_coder_%j.log
#SBATCH --error=logs/qwen_coder_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32

# Test configurations for Qwen3-Coder-480B model
# This script tests with 8 H100 GPUs

echo "=============================================="
echo "Testing Qwen3-Coder-480B-A35B-Instruct-FP8"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8x H100"
echo "=============================================="

# Load modules
module load cuda/12.1
module load python/3.10

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nl2sql

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=32
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

# Test coding-specific prompts
PROMPTS=(
    "Write a SQL query to find the top 5 customers by total purchase amount from orders and customers tables."
    "Implement a Python class for a binary search tree with insert, search, and delete operations."
    "Create a function to parse and validate JSON schema against a given JSON object."
)

# Run tests
for i in "${!PROMPTS[@]}"; do
    echo "Test $((i+1)): ${PROMPTS[$i]}"

    python scripts/test_qwen_hf.py \
        --model coder \
        --num_gpus 8 \
        --max_tokens 2048 \
        --prompt "${PROMPTS[$i]}" \
        --output_file "results/coder_test_${SLURM_JOB_ID}_prompt_$i.json"

    echo "Test $((i+1)) completed"
    echo "---"
done

echo "All tests completed for Qwen3-Coder-480B"