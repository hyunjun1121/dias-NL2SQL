#!/usr/bin/env python
"""Check environment for MiniMax-M2 requirements."""

import sys
import subprocess

print("Checking MiniMax-M2 Requirements...")
print("="*50)

# Python version
python_version = sys.version_info
print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
if 3.9 <= python_version.major + python_version.minor/10 <= 3.12:
    print("  ✓ Python version OK (3.9-3.12)")
else:
    print("  ✗ Python version should be 3.9-3.12")

# Transformers version
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
    if transformers.__version__ >= "4.57.1":
        print("  ✓ Transformers version OK")
    else:
        print(f"  ⚠️ Recommended: 4.57.1+")
except ImportError:
    print("  ✗ Transformers not installed")

# PyTorch and CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            if props.major >= 7:
                print("    ✓ Compute capability OK (7.0+)")
            else:
                print("    ✗ Need compute capability 7.0+")
except ImportError:
    print("  ✗ PyTorch not installed")

# Accelerate
try:
    import accelerate
    print(f"Accelerate: {accelerate.__version__}")
    print("  ✓ Accelerate installed")
except ImportError:
    print("  ⚠️ Accelerate not installed (optional but recommended)")

print("\n" + "="*50)