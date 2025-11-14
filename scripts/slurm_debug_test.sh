#!/bin/bash
#SBATCH --job-name=qwen_debug_test
#SBATCH --output=logs/qwen_debug_%j.log
#SBATCH --error=logs/qwen_debug_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=16

# Debug test with single sample and detailed logging
# Tests with minimal GPU allocation first

echo "=========================================="
echo "Qwen Debug Test - Single Sample"
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

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export HF_HOME=/scratch/$USER/huggingface_cache
export CUDA_LAUNCH_BLOCKING=1

# Create directories
mkdir -p logs
mkdir -p debug_logs_thinking
mkdir -p debug_logs_coder
mkdir -p $TRANSFORMERS_CACHE

# Print system information
echo "System Information:"
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# Test 1: With Ollama (baseline)
echo "========== Test 1: Ollama Baseline =========="
python scripts/test_qwen_debug.py \
    --model thinking \
    --use_ollama

echo "Ollama test completed"
echo ""

# Test 2: With HuggingFace Thinking model
echo "========== Test 2: HF Thinking Model =========="
python scripts/test_qwen_debug.py \
    --model thinking \
    --use_hf

echo "HF Thinking model test completed"
echo ""

# Generate summary report
echo "========== Generating Summary Report =========="
python -c "
import json
import glob
from pathlib import Path

print('\\nDebug Test Summary')
print('='*60)

# Check for debug logs
for model_type in ['thinking', 'coder']:
    log_dir = Path(f'debug_logs_{model_type}')
    if log_dir.exists():
        session_logs = list(log_dir.glob('session_*.json'))
        if session_logs:
            latest_session = max(session_logs, key=lambda p: p.stat().st_mtime)
            print(f'\\n{model_type.upper()} Model:')
            print(f'  Latest session: {latest_session.name}')

            with open(latest_session, 'r') as f:
                data = json.load(f)
                print(f'  Total LLM calls: {data.get(\"total_calls\", 0)}')

                # Count calls by type
                if 'logs' in data:
                    prompt_lengths = []
                    response_lengths = []
                    for log_entry in data['logs']:
                        if 'input' in log_entry:
                            prompt_lengths.append(len(log_entry['input'].get('prompt', '')))
                        if 'output' in log_entry:
                            response_lengths.append(len(log_entry['output'].get('response', '')))

                    if prompt_lengths:
                        print(f'  Avg prompt length: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars')
                    if response_lengths:
                        print(f'  Avg response length: {sum(response_lengths)/len(response_lengths):.0f} chars')

            # Check pipeline results
            result_files = list(log_dir.glob('pipeline_results_*.json'))
            if result_files:
                latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
                with open(latest_result, 'r') as f:
                    results = json.load(f)
                    print(f'  Pipeline steps:')
                    for step, result in results.get('steps', {}).items():
                        status = '✓' if result.get('success') else '✗'
                        print(f'    {status} {step}')

print('\\n' + '='*60)
"

echo ""
echo "All debug tests completed"
echo "Check debug_logs_thinking/ and debug_logs_coder/ for detailed logs"