# Qwen Large Models Setup Guide

## Quick Start

### 1. Local Development Setup
```bash
# Create and setup conda environment
bash setup_conda_env.sh

# Activate environment
conda activate nl2sql_qwen

# Install flash attention (optional, for better performance)
bash install_flash_attn.sh
```

### 2. HPC Server Setup
```bash
# SSH to server
ssh username@hpc-server

# Clone repository
git clone https://github.com/your-repo/nl2sql-baseline.git
cd nl2sql-baseline/dias-NL2SQL

# Run server setup
bash setup_server_env.sh

# Load modules and activate environment
module load cuda/12.1 python/3.10
conda activate nl2sql_qwen
```

## Environment Structure

### Conda Environment: `nl2sql_qwen`
- Python 3.10
- PyTorch 2.1+ with CUDA 12.1
- Transformers 4.51+ (required for Qwen3 MoE)
- Flash Attention 2.5+ (optional, recommended)

### Key Packages
| Package | Version | Purpose |
|---------|---------|---------|
| transformers | ≥4.51.0 | Qwen3 MoE model support |
| torch | ≥2.1.0 | Deep learning framework |
| flash-attn | ≥2.5.0 | Optimized attention (optional) |
| accelerate | ≥0.25.0 | Multi-GPU support |
| ollama | ≥0.1.0 | Fallback LLM client |

## Directory Structure

```
dias-NL2SQL/
├── setup_conda_env.sh        # Local conda setup
├── setup_server_env.sh        # HPC server setup
├── requirements_qwen.txt      # Python dependencies
├── install_flash_attn.sh      # Flash attention installer
├── scripts/
│   ├── test_qwen_debug.py    # Debug test with logging
│   ├── simple_hf_test.py     # Minimal model test
│   ├── slurm_*.sh             # SLURM job scripts
│   └── estimate_gpu_requirements.py
├── utils/
│   ├── hf_llm_client.py      # HuggingFace client
│   └── llm_client.py          # Ollama client
└── debug_logs_*/              # Debug output directories
```

## Testing Workflow

### Step 1: Verify Environment
```bash
# Check GPU availability
nvidia-smi

# Test Python packages
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Version: {transformers.__version__}')"
```

### Step 2: Run Simple Test
```bash
# Local test (without SLURM)
python scripts/simple_hf_test.py --model thinking --save_results

# On HPC with SLURM
sbatch scripts/slurm_simple_test.sh
```

### Step 3: Run Debug Test
```bash
# Test with Ollama first (baseline)
python scripts/test_qwen_debug.py --model thinking --use_ollama

# Test with HuggingFace model
python scripts/test_qwen_debug.py --model thinking --use_hf

# Submit to SLURM
sbatch scripts/slurm_debug_test.sh
```

### Step 4: Check Results
```bash
# View logs
tail -f logs/qwen_debug_*.log

# Check debug outputs
ls -la debug_logs_thinking/
cat debug_logs_thinking/session_*.json | head -100
```

## Model Configurations

### Qwen3-235B-A22B-Thinking
- **Min GPUs**: 4x H100 (80GB each)
- **Recommended**: 8x H100
- **Memory**: ~235GB total
- **Use case**: Complex reasoning tasks

### Qwen3-Coder-480B-A35B
- **Min GPUs**: 8x H100 (80GB each)
- **Recommended**: 16x H100
- **Memory**: ~480GB total
- **Use case**: Code generation

## Environment Variables

```bash
# Required for HuggingFace
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export HF_HOME=/scratch/$USER/huggingface_cache

# For distributed inference
export CUDA_LAUNCH_BLOCKING=1  # For FP8 models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional optimizations
export OMP_NUM_THREADS=16
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)
```bash
# Solution: Reduce max tokens or increase GPUs
python scripts/test_qwen_debug.py --max_tokens 512
```

### Issue 2: Transformers version error
```bash
# Error: KeyError: 'qwen3_moe'
# Solution: Upgrade transformers
pip install transformers>=4.51.0
```

### Issue 3: Flash attention compilation fails
```bash
# Solution: Use pre-built wheels or skip
pip install flash-attn --no-build-isolation
# Or continue without it (slower but works)
```

### Issue 4: CUDA version mismatch
```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch if needed
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Performance Tips

1. **Use Flash Attention**: 2-3x faster inference
2. **Set proper batch size**: Start with 1, increase gradually
3. **Monitor GPU memory**: Use `nvidia-smi -l 1`
4. **Use tensor parallelism**: Distribute model across GPUs
5. **Cache models locally**: Avoid re-downloading

## Monitoring

### During execution
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor Python process
htop -p $(pgrep -f python)

# Check memory usage
free -h
```

### After execution
```bash
# Analyze logs
grep "Error\|Warning" logs/*.log

# Check debug outputs
python -c "
import json
with open('debug_logs_thinking/session_XXX.json') as f:
    data = json.load(f)
    print(f'Total calls: {data[\"total_calls\"]}')
    for log in data['logs']:
        print(f'Call {log[\"call_id\"]}: {log[\"output\"][\"elapsed_time\"]:.2f}s')
"
```

## Support

For issues:
1. Check error logs in `logs/` directory
2. Review debug outputs in `debug_logs_*/`
3. Verify environment with setup scripts
4. Contact HPC support for resource issues

---

*Last updated: 2025*
*Tested with: transformers 4.51.0, torch 2.1.0, CUDA 12.1*