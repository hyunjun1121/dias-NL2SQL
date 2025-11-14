#!/bin/bash
#SBATCH --job-name=qwen_simple_test
#SBATCH --output=logs/simple_test_%j.log
#SBATCH --error=logs/simple_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --cpus-per-task=16

# Simple minimal test for Qwen models
# Just verify model loading and basic generation

echo "=========================================="
echo "Simple Qwen Model Test"
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

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export HF_HOME=/scratch/$USER/huggingface_cache
export CUDA_LAUNCH_BLOCKING=1

# Create directories
mkdir -p logs
mkdir -p results
mkdir -p $TRANSFORMERS_CACHE

# System info
echo "System Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# Run simple test
echo "========== Running Simple Test =========="
python scripts/simple_hf_test.py --model thinking --save_results

echo ""
echo "Test completed successfully!"

# Check if output was saved
if ls simple_test_*.json 1> /dev/null 2>&1; then
    echo "Results saved:"
    ls -la simple_test_*.json
    echo ""
    echo "Result preview:"
    head -20 simple_test_*.json
fi