#!/bin/bash
# Special installation script for flash-attention
# This requires compilation and can be tricky on some systems

echo "=========================================="
echo "Installing Flash Attention for Qwen Models"
echo "=========================================="

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    echo "Try: module load cuda/12.1"
    exit 1
fi

echo "CUDA version:"
nvcc --version

# Check Python and pip
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please activate your conda environment."
    exit 1
fi

echo "Python version:"
python --version

# Install ninja for faster compilation
echo "Installing build tools..."
pip install ninja packaging

# Set environment variables for compilation
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # For A100, H100
export MAX_JOBS=8  # Limit parallel compilation jobs

# Option 1: Try pre-built wheel first (faster)
echo "Attempting to install pre-built flash-attn..."
pip install flash-attn --no-build-isolation

if [ $? -ne 0 ]; then
    echo "Pre-built installation failed. Trying from source..."

    # Option 2: Build from source
    echo "Installing from source (this will take 10-30 minutes)..."

    # Clone the repository
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention

    # Checkout stable version
    git checkout v2.5.0

    # Install
    python setup.py install

    cd ..
fi

# Verify installation
echo ""
echo "Verifying flash-attention installation..."
python -c "
try:
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__}')
    print('Flash Attention installed successfully!')
except ImportError as e:
    print(f'Flash Attention not installed: {e}')
    print('This is optional - models will work without it, just slower.')
"

# Alternative: Install xFormers (easier to install)
echo ""
echo "Installing xFormers as alternative/complement..."
pip install xformers

python -c "
try:
    import xformers
    print(f'xFormers version: {xformers.__version__}')
    print('xFormers installed successfully!')
except ImportError:
    print('xFormers not installed.')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Note: Flash attention is optional but recommended for performance."
echo "Models will automatically use it if available."
echo "=========================================="