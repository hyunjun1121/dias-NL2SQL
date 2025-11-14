#!/bin/bash
# Setup conda environment for Qwen large models testing

echo "=========================================="
echo "Setting up Conda Environment for NL2SQL with Qwen Models"
echo "=========================================="

# Environment name
ENV_NAME="nl2sql_qwen"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create new environment with Python 3.10
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install transformers and related packages
echo "Installing HuggingFace packages..."
pip install --upgrade transformers accelerate datasets
pip install flash-attn --no-build-isolation  # For flash attention support

# Install specific version requirements for Qwen models
echo "Installing Qwen model requirements..."
pip install transformers>=4.51.0  # Required for Qwen3 MoE
pip install sentencepiece protobuf  # Tokenizer requirements
pip install einops  # For model operations

# Install NL2SQL pipeline dependencies
echo "Installing NL2SQL dependencies..."
pip install -r requirements.txt

# Install additional packages for large model inference
echo "Installing optimization packages..."
pip install bitsandbytes  # For quantization options
pip install optimum  # HuggingFace optimization library
pip install peft  # Parameter-efficient fine-tuning (optional)

# Install monitoring and debugging tools
echo "Installing monitoring tools..."
pip install gpustat
pip install psutil
pip install py-cpuinfo

# Install Ollama client (as fallback)
echo "Installing Ollama client..."
pip install ollama

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"