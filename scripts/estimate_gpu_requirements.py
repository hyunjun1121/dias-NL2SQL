"""
GPU requirement estimation for Qwen models.
Provides detailed memory calculations and node allocation recommendations.
"""

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
import argparse

@dataclass
class ModelConfig:
    """Model configuration details."""
    name: str
    total_params_b: float  # Billions
    active_params_b: float  # Billions (for MoE)
    num_experts: int
    active_experts: int
    quantization: str  # FP8, FP16, etc.
    context_length: int

@dataclass
class GPUSpec:
    """GPU specifications."""
    name: str
    memory_gb: int
    fp8_tflops: float
    fp16_tflops: float

@dataclass
class RequirementEstimate:
    """Estimated requirements for running a model."""
    model_name: str
    min_gpus: int
    recommended_gpus: int
    optimal_gpus: int
    memory_per_gpu_gb: float
    total_memory_gb: float
    tensor_parallel_degree: int
    pipeline_parallel_degree: int
    estimated_tps: float  # tokens per second
    notes: List[str]

# Model configurations
MODELS = {
    "thinking": ModelConfig(
        name="Qwen3-235B-A22B-Thinking-2507-FP8",
        total_params_b=235,
        active_params_b=22,
        num_experts=128,
        active_experts=8,
        quantization="FP8",
        context_length=262144
    ),
    "coder": ModelConfig(
        name="Qwen3-Coder-480B-A35B-Instruct-FP8",
        total_params_b=480,
        active_params_b=35,
        num_experts=160,
        active_experts=8,
        quantization="FP8",
        context_length=262144
    )
}

# GPU specifications
GPUS = {
    "H100": GPUSpec(
        name="NVIDIA H100 80GB",
        memory_gb=80,
        fp8_tflops=1979,
        fp16_tflops=989
    ),
    "A100": GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80,
        fp8_tflops=624,
        fp16_tflops=312
    ),
    "A100_40": GPUSpec(
        name="NVIDIA A100 40GB",
        memory_gb=40,
        fp8_tflops=624,
        fp16_tflops=312
    )
}

def calculate_memory_requirements(model: ModelConfig) -> Dict:
    """Calculate memory requirements for a model."""

    # Base memory calculations
    if model.quantization == "FP8":
        bytes_per_param = 1
    elif model.quantization == "FP16":
        bytes_per_param = 2
    else:  # FP32
        bytes_per_param = 4

    # Model weights memory
    total_params = model.total_params_b * 1e9
    model_memory_gb = (total_params * bytes_per_param) / (1024**3)

    # KV cache memory (rough estimate)
    # Per token: 2 * num_layers * hidden_dim * bytes_per_param
    # Estimate: ~0.1 GB per 1K tokens for large models
    kv_cache_per_1k_tokens = 0.1
    max_context_kv_cache_gb = (model.context_length / 1000) * kv_cache_per_1k_tokens

    # Activation memory (for active parameters)
    activation_memory_gb = model.active_params_b * 0.5  # Rough estimate

    # Overhead (optimizer states, gradients, etc. - for inference only)
    overhead_factor = 1.2  # 20% overhead

    total_memory_gb = (model_memory_gb + max_context_kv_cache_gb + activation_memory_gb) * overhead_factor

    return {
        "model_memory_gb": model_memory_gb,
        "kv_cache_gb": max_context_kv_cache_gb,
        "activation_gb": activation_memory_gb,
        "total_memory_gb": total_memory_gb,
        "overhead_factor": overhead_factor
    }

def calculate_gpu_allocation(model: ModelConfig, gpu: GPUSpec) -> RequirementEstimate:
    """Calculate optimal GPU allocation for a model."""

    memory_reqs = calculate_memory_requirements(model)
    total_memory = memory_reqs["total_memory_gb"]

    # Calculate minimum GPUs needed
    min_gpus = math.ceil(total_memory / (gpu.memory_gb * 0.95))  # 95% usable memory

    # For MoE models, we want at least 1 GPU per active expert for efficiency
    moe_optimal_gpus = model.active_experts

    # Recommended: balance between memory and compute
    recommended_gpus = max(min_gpus, min(moe_optimal_gpus, min_gpus * 2))

    # Optimal: best performance
    optimal_gpus = max(recommended_gpus, moe_optimal_gpus)

    # Ensure power of 2 for better parallelization
    for gpu_count in [min_gpus, recommended_gpus, optimal_gpus]:
        if gpu_count > 1:
            gpu_count = 2 ** math.ceil(math.log2(gpu_count))

    # Calculate parallelization strategy
    if optimal_gpus <= 8:
        tensor_parallel = optimal_gpus
        pipeline_parallel = 1
    else:
        tensor_parallel = 8
        pipeline_parallel = optimal_gpus // 8

    # Estimate tokens per second (rough approximation)
    # Based on active parameters and GPU compute
    if model.quantization == "FP8":
        compute_tflops = gpu.fp8_tflops * optimal_gpus
    else:
        compute_tflops = gpu.fp16_tflops * optimal_gpus

    # Very rough estimate: 1 TFLOP ≈ 10 tokens/sec for 1B params
    estimated_tps = (compute_tflops / model.active_params_b) * 10

    notes = []
    if min_gpus < moe_optimal_gpus:
        notes.append(f"Model has {model.num_experts} experts ({model.active_experts} active), optimal to have {moe_optimal_gpus} GPUs")
    if total_memory > gpu.memory_gb * min_gpus * 0.8:
        notes.append("Memory usage is tight, consider using more GPUs for better performance")
    if model.context_length > 100000:
        notes.append(f"Long context ({model.context_length} tokens) will require significant KV cache memory")

    return RequirementEstimate(
        model_name=model.name,
        min_gpus=min_gpus,
        recommended_gpus=recommended_gpus,
        optimal_gpus=optimal_gpus,
        memory_per_gpu_gb=total_memory / optimal_gpus,
        total_memory_gb=total_memory,
        tensor_parallel_degree=tensor_parallel,
        pipeline_parallel_degree=pipeline_parallel,
        estimated_tps=estimated_tps,
        notes=notes
    )

def generate_slurm_script(model: ModelConfig, gpu: GPUSpec, num_gpus: int) -> str:
    """Generate SLURM script for specific configuration."""

    # Calculate resources
    mem_per_gpu = int(gpu.memory_gb * 0.9)
    total_mem = num_gpus * mem_per_gpu
    cpus_per_gpu = 8

    script = f"""#!/bin/bash
#SBATCH --job-name={model.name.split('-')[0].lower()}_test
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --mem={total_mem}G
#SBATCH --cpus-per-task={num_gpus * cpus_per_gpu}

# Auto-generated SLURM script for {model.name}
# GPUs: {num_gpus}x {gpu.name}

module load cuda/12.1
module load python/3.10

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nl2sql

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 {num_gpus-1})
export OMP_NUM_THREADS={num_gpus * cpus_per_gpu}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export CUDA_LAUNCH_BLOCKING=1

python scripts/test_qwen_hf.py \\
    --model {model.name.split('-')[0].lower()} \\
    --num_gpus {num_gpus} \\
    --max_tokens 1024 \\
    --output_file results/test_${{SLURM_JOB_ID}}.json
"""
    return script

def print_recommendations(estimate: RequirementEstimate, gpu: GPUSpec):
    """Print formatted recommendations."""

    print(f"\n{'='*60}")
    print(f"Model: {estimate.model_name}")
    print(f"GPU: {gpu.name}")
    print(f"{'='*60}")

    print(f"\nMemory Requirements:")
    print(f"  Total Model Memory: {estimate.total_memory_gb:.1f} GB")
    print(f"  Memory per GPU: {estimate.memory_per_gpu_gb:.1f} GB")

    print(f"\nGPU Allocation:")
    print(f"  Minimum GPUs: {estimate.min_gpus} (may be slow)")
    print(f"  Recommended: {estimate.recommended_gpus} (balanced)")
    print(f"  Optimal: {estimate.optimal_gpus} (best performance)")

    print(f"\nParallelization Strategy:")
    print(f"  Tensor Parallel: {estimate.tensor_parallel_degree}")
    print(f"  Pipeline Parallel: {estimate.pipeline_parallel_degree}")

    print(f"\nEstimated Performance:")
    print(f"  Tokens/Second: ~{estimate.estimated_tps:.0f}")

    if estimate.notes:
        print(f"\nNotes:")
        for note in estimate.notes:
            print(f"  - {note}")

    # Node allocation
    nodes_needed = math.ceil(estimate.optimal_gpus / 8)  # Assuming 8 GPUs per node
    print(f"\nNode Allocation:")
    print(f"  Nodes needed: {nodes_needed}")
    print(f"  Total GPUs: {estimate.optimal_gpus}")

def main():
    parser = argparse.ArgumentParser(description="Estimate GPU requirements for Qwen models")
    parser.add_argument("--model", choices=["thinking", "coder", "both"], default="both",
                      help="Which model to analyze")
    parser.add_argument("--gpu", choices=["H100", "A100", "A100_40"], default="H100",
                      help="GPU type")
    parser.add_argument("--generate_slurm", action="store_true",
                      help="Generate SLURM scripts")
    parser.add_argument("--compare_gpus", action="store_true",
                      help="Compare different GPU types")

    args = parser.parse_args()

    models_to_analyze = []
    if args.model == "both":
        models_to_analyze = ["thinking", "coder"]
    else:
        models_to_analyze = [args.model]

    if args.compare_gpus:
        # Compare all GPU types
        for model_key in models_to_analyze:
            model = MODELS[model_key]
            print(f"\n\n{'#'*70}")
            print(f"# {model.name}")
            print(f"{'#'*70}")

            for gpu_key, gpu in GPUS.items():
                estimate = calculate_gpu_allocation(model, gpu)
                print_recommendations(estimate, gpu)

    else:
        # Single GPU type analysis
        gpu = GPUS[args.gpu]

        for model_key in models_to_analyze:
            model = MODELS[model_key]
            estimate = calculate_gpu_allocation(model, gpu)
            print_recommendations(estimate, gpu)

            if args.generate_slurm:
                # Generate SLURM scripts for different configurations
                for num_gpus in [estimate.min_gpus, estimate.recommended_gpus, estimate.optimal_gpus]:
                    filename = f"slurm_{model_key}_{num_gpus}gpu.sh"
                    script = generate_slurm_script(model, gpu, num_gpus)

                    with open(filename, 'w') as f:
                        f.write(script)

                    print(f"\nGenerated SLURM script: {filename}")

    # Summary comparison table
    if args.model == "both":
        print(f"\n\n{'='*70}")
        print("Summary Comparison (using {})".format(GPUS[args.gpu].name))
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Min GPUs':<10} {'Rec GPUs':<10} {'Memory (GB)':<12} {'Est. TPS':<10}")
        print("-" * 70)

        for model_key in ["thinking", "coder"]:
            model = MODELS[model_key]
            estimate = calculate_gpu_allocation(model, GPUS[args.gpu])
            print(f"{model.name[:28]:<30} {estimate.min_gpus:<10} {estimate.recommended_gpus:<10} "
                  f"{estimate.total_memory_gb:<12.1f} {estimate.estimated_tps:<10.0f}")

if __name__ == "__main__":
    main()