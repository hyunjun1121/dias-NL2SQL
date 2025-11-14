#!/bin/bash
# Quick conda setup - minimal installation for testing

echo "=========================================="
echo "Quick Conda Setup (Minimal)"
echo "=========================================="

ENV_NAME="qwen_test"

# Create environment if not exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating $ENV_NAME environment..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Active environment: $CONDA_DEFAULT_ENV"

# Install only essential packages
echo "Installing essential packages only..."

# Core packages - install one by one to catch errors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing transformers..."
pip install "transformers>=4.51.0"

echo "Installing accelerate..."
pip install accelerate

echo "Installing basic dependencies..."
pip install sentencepiece protobuf tokenizers einops safetensors

echo "Installing NL2SQL basics..."
pip install pandas sqlalchemy numpy ollama

echo "Installing langchain..."
pip install langchain langchain-community langchain-core

echo "Installing utilities..."
pip install tqdm psutil

# Quick verification
echo ""
echo "Quick verification:"
python -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

echo ""
echo "Setup complete! Activate with: conda activate $ENV_NAME"