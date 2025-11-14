#!/bin/bash
# Server-specific setup for HPC cluster with module system

echo "=========================================="
echo "HPC Server Environment Setup for Qwen Models"
echo "=========================================="

# Check if we're on a cluster with module system
if command -v module &> /dev/null; then
    echo "Module system detected. Loading required modules..."

    # Load required modules (adjust versions as needed)
    module purge  # Clean slate
    module load gcc/11.3.0
    module load cuda/12.1
    module load cudnn/8.9.0
    module load python/3.10
    module load git/2.40.0
    module load git-lfs/3.3.0

    echo "Loaded modules:"
    module list
else
    echo "No module system detected. Using system defaults."
fi

# Set up paths
export PROJECT_DIR="$HOME/nl2sql-baseline/dias-NL2SQL"
export SCRATCH_DIR="/scratch/$USER"
export CACHE_DIR="$SCRATCH_DIR/huggingface_cache"
export DATA_DIR="$PROJECT_DIR/data"

# Create necessary directories
echo "Creating directories..."
mkdir -p $CACHE_DIR
mkdir -p $SCRATCH_DIR/models
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/results
mkdir -p $PROJECT_DIR/debug_logs_thinking
mkdir -p $PROJECT_DIR/debug_logs_coder

# Set environment variables for HuggingFace
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TORCH_HOME=$CACHE_DIR/torch

echo "Cache directories set to: $CACHE_DIR"

# Check for existing conda installation
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Warning: Conda not found in standard locations."
    echo "Please install Miniconda or adjust paths."
fi

# Create or update conda environment
ENV_NAME="nl2sql_qwen"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME exists. Activating..."
    conda activate $ENV_NAME
else
    echo "Creating new environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME

    # Install packages
    echo "Installing packages..."

    # PyTorch with CUDA
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

    # Install from requirements
    pip install -r $PROJECT_DIR/requirements_qwen.txt
fi

# Download models (optional - can be done on first run)
echo ""
echo "Model download setup:"
echo "Models will be downloaded to: $CACHE_DIR"
echo "To pre-download models, run:"
echo "  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-235B-A22B-Thinking-2507-FP8')\""

# Set up Git LFS for large files
if command -v git-lfs &> /dev/null; then
    echo "Initializing Git LFS..."
    cd $PROJECT_DIR
    git lfs install
    git lfs fetch --all
fi

# Verify GPU availability
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not found. GPU status unknown."
fi

# Test Python environment
echo ""
echo "Testing Python environment..."
python << EOF
import sys
import torch
import transformers

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To use this environment:"
echo "  1. Load modules (if on HPC): module load cuda/12.1 python/3.10"
echo "  2. Activate conda: conda activate $ENV_NAME"
echo "  3. Set cache: export TRANSFORMERS_CACHE=$CACHE_DIR"
echo ""
echo "To run tests:"
echo "  - Simple test: sbatch scripts/slurm_simple_test.sh"
echo "  - Debug test: sbatch scripts/slurm_debug_test.sh"
echo "  - Full test: sbatch scripts/slurm_test_thinking.sh"
echo "=========================================="