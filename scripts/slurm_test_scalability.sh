#!/bin/bash
#SBATCH --job-name=qwen_scalability
#SBATCH --output=logs/qwen_scale_%j.log
#SBATCH --error=logs/qwen_scale_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32

# Scalability test - try different GPU configurations
# This script tests how models perform with different GPU counts

echo "=============================================="
echo "Qwen Scalability Testing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="

# Load modules
module load cuda/12.1
module load python/3.10

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nl2sql

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export HF_HOME=/scratch/$USER/huggingface_cache
export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs
mkdir -p results
mkdir -p $TRANSFORMERS_CACHE

# Test prompt
TEST_PROMPT="Generate a SQL query to find employees with salary above department average."

# Test Qwen3-235B-A22B-Thinking with different GPU counts
echo "========== Testing Qwen3-235B-A22B-Thinking =========="

# Test with 2 GPUs (will likely fail - testing minimum)
echo "Testing with 2 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1
python scripts/test_qwen_hf.py \
    --model thinking \
    --num_gpus 2 \
    --max_tokens 512 \
    --prompt "$TEST_PROMPT" \
    --output_file "results/scale_thinking_2gpu_${SLURM_JOB_ID}.json" || echo "2 GPU test failed (expected)"

# Test with 4 GPUs (minimum recommended)
echo "Testing with 4 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/test_qwen_hf.py \
    --model thinking \
    --num_gpus 4 \
    --max_tokens 512 \
    --prompt "$TEST_PROMPT" \
    --output_file "results/scale_thinking_4gpu_${SLURM_JOB_ID}.json"

# Test with 8 GPUs (optimal)
echo "Testing with 8 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python scripts/test_qwen_hf.py \
    --model thinking \
    --num_gpus 8 \
    --max_tokens 512 \
    --prompt "$TEST_PROMPT" \
    --output_file "results/scale_thinking_8gpu_${SLURM_JOB_ID}.json"

echo ""
echo "========== Testing Qwen3-Coder-480B =========="

# Test with 4 GPUs (will likely fail - testing minimum)
echo "Testing with 4 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/test_qwen_hf.py \
    --model coder \
    --num_gpus 4 \
    --max_tokens 512 \
    --prompt "$TEST_PROMPT" \
    --output_file "results/scale_coder_4gpu_${SLURM_JOB_ID}.json" || echo "4 GPU test failed (expected)"

# Test with 8 GPUs (minimum recommended)
echo "Testing with 8 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python scripts/test_qwen_hf.py \
    --model coder \
    --num_gpus 8 \
    --max_tokens 512 \
    --prompt "$TEST_PROMPT" \
    --output_file "results/scale_coder_8gpu_${SLURM_JOB_ID}.json"

echo ""
echo "Scalability testing completed"

# Generate summary report
python -c "
import json
import glob

print('\\n========== Scalability Test Summary ==========')
for result_file in sorted(glob.glob('results/scale_*_${SLURM_JOB_ID}.json')):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)

        model = 'Thinking' if 'thinking' in result_file else 'Coder'
        gpus = result_file.split('_')[-2].replace('gpu', '')

        print(f'\\n{model} Model with {gpus} GPUs:')
        print(f'  Success: {data[\"success\"]}')
        if data['success']:
            print(f'  Loading Time: {data[\"loading_time\"]:.2f}s')
            print(f'  Inference Time: {data[\"inference_time\"]:.2f}s')
            print(f'  Tokens/Second: {data[\"tokens_per_second\"]:.2f}')
            if 'gpu_memory' in data and data['gpu_memory']:
                final = data['gpu_memory'].get('after_inference', {})
                if final:
                    print(f'  GPU Memory: {final.get(\"allocated_gb\", 0):.2f} GB allocated')
        else:
            print(f'  Error: {data.get(\"error\", \"Unknown\")}')
    except Exception as e:
        print(f'Could not read {result_file}: {e}')
"