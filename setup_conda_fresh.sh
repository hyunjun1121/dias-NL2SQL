#!/bin/bash
# Fresh conda environment setup for Qwen models on HPC

echo "=========================================="
echo "Fresh Conda Environment Setup for Qwen Models"
echo "=========================================="

# Environment name
ENV_NAME="qwen_nl2sql"

# Check current environment
echo "Current conda environments:"
conda env list
echo ""

# Remove existing environment if exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing $ENV_NAME environment..."
    conda deactivate 2>/dev/null
    conda env remove -n $ENV_NAME -y
fi

# Create fresh environment
echo "Creating new conda environment: $ENV_NAME with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Verify activation
echo "Active environment: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Error: Failed to activate $ENV_NAME"
    exit 1
fi

# Install PyTorch with CUDA 12.1
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "Error: PyTorch installation failed"
    exit 1
fi

# Install essential packages first
echo ""
echo "Installing essential packages..."
pip install --upgrade pip setuptools wheel

# Install transformers with specific version for Qwen3 MoE
echo "Installing HuggingFace transformers (>=4.51.0 for Qwen3)..."
pip install "transformers>=4.51.0"

# Install accelerate for multi-GPU support
echo "Installing accelerate..."
pip install "accelerate>=0.25.0"

# Install tokenizers and related
echo "Installing tokenizer packages..."
pip install tokenizers sentencepiece protobuf

# Install model-specific packages
echo "Installing model-specific packages..."
pip install einops safetensors

# Install NL2SQL dependencies
echo "Installing NL2SQL pipeline dependencies..."
pip install sqlalchemy pandas "numpy<2.0.0"
pip install langchain langchain-community langchain-core

# Install Ollama client
echo "Installing Ollama client..."
pip install ollama

# Install database and utility packages
echo "Installing utility packages..."
pip install datasketch chromadb func-timeout
pip install psutil tqdm

# Install optimization packages (optional but recommended)
echo "Installing optimization packages..."
pip install optimum
# Try to install bitsandbytes (may fail on some systems)
pip install bitsandbytes || echo "bitsandbytes installation failed (optional)"

# Monitoring tools
echo "Installing monitoring tools..."
pip install gpustat py-cpuinfo

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p logs
mkdir -p results
mkdir -p debug_logs_thinking
mkdir -p debug_logs_coder

# Verify all critical imports
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
    # Check version is sufficient for Qwen3
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.51.0"):
        print("  Version OK for Qwen3 MoE")
    else:
        print("  WARNING: Version may be too old for Qwen3 MoE")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import accelerate
    print(f"✓ Accelerate: {accelerate.__version__}")
except ImportError as e:
    print(f"✗ Accelerate: {e}")

try:
    import ollama
    print(f"✓ Ollama: installed")
except ImportError as e:
    print(f"✗ Ollama: {e}")

try:
    import langchain
    print(f"✓ LangChain: {langchain.__version__}")
except ImportError as e:
    print(f"✗ LangChain: {e}")

try:
    from utils.llm_client import LLMClient
    print(f"✓ LLMClient: importable")
except ImportError as e:
    print(f"✗ LLMClient: {e}")

print("\n" + "="*40)
print("Environment setup complete!")
print("="*40)
EOF

# Save environment info
echo ""
echo "Saving environment information..."
conda list > "conda_env_${ENV_NAME}_$(date +%Y%m%d).txt"
pip freeze > "pip_freeze_${ENV_NAME}_$(date +%Y%m%d).txt"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: $ENV_NAME"
echo "Python: $(python --version)"
echo ""
echo "To activate this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the setup:"
echo "  python scripts/test_qwen_debug.py --mock"
echo ""
echo "Environment details saved to:"
echo "  conda_env_${ENV_NAME}_$(date +%Y%m%d).txt"
echo "  pip_freeze_${ENV_NAME}_$(date +%Y%m%d).txt"
echo "=========================================="