#!/usr/bin/env python
"""Compare Qwen3 Thinking, Qwen3 Coder, and MiniMax-M2 models."""

import json
import os

print("="*80)
print("Model Comparison: Qwen3-Thinking vs Qwen3-Coder vs MiniMax-M2")
print("="*80)
print()

models = {
    "Qwen3-235B-Thinking": {
        "full_name": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        "params": "235B total, 22B active",
        "memory_tested": 220.31,
        "gpus_required": 4,
        "context": "262K tokens",
        "thinking_mode": "Yes (<think> tags)",
        "best_for": "Complex reasoning, multi-step analysis",
        "status": "✅ Tested on 4xH100"
    },
    "Qwen3-480B-Coder": {
        "full_name": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "params": "480B total, 35B active",
        "memory_estimated": 480,
        "gpus_required": 6,
        "context": "256K tokens (1M with Yarn)",
        "thinking_mode": "No",
        "best_for": "Code generation, tool calling, agentic tasks",
        "status": "⚠️ Needs 6+ GPUs"
    },
    "MiniMax-M2": {
        "full_name": "MiniMaxAI/MiniMax-M2",
        "params": "~220GB weights",
        "memory_estimated": 220,
        "gpus_required": 4,
        "context": "TBD",
        "thinking_mode": "Yes (interleaved <think> tags)",
        "best_for": "General reasoning with thinking",
        "status": "🔄 Testing pending"
    }
}

# Print comparison table
print("📊 Model Specifications")
print("-" * 80)
for name, specs in models.items():
    print(f"\n{name}:")
    for key, value in specs.items():
        if key != "full_name":
            print(f"  {key.replace('_', ' ').title()}: {value}")

print("\n" + "="*80)
print("🎯 Recommendations for NL2SQL Pipeline")
print("-" * 80)

print("""
1. For Query Understanding & Planning:
   • Primary: Qwen3-235B-Thinking (4 GPUs, tested ✓)
   • Alternative: MiniMax-M2 (4 GPUs, similar memory)

2. For SQL Generation & Optimization:
   • Ideal: Qwen3-480B-Coder (needs 6+ GPUs)
   • Current: Use Qwen3-235B-Thinking

3. Deployment Strategy:
   • Development: 4 GPUs with Thinking model
   • Production: 6-8 GPUs for both models
   • Fallback: MiniMax-M2 as alternative

4. Context Window Considerations:
   • Large schemas: Qwen3 models (256K+ tokens)
   • Standard queries: Any model
   • Repository-scale: Qwen3-Coder (1M tokens)
""")

# Check for test results
print("\n📁 Test Results:")
print("-" * 80)

result_files = {
    "Qwen3-235B": "results/final_gpu_test_results.json",
    "Qwen3-Coder": "results/qwen3_coder_test.json",
    "MiniMax-M2": "results/minimax_m2_official_test.json"
}

for model, filepath in result_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ {model}: Found test results")
        if 'memory_used_gb' in data:
            print(f"  Memory used: {data['memory_used_gb']:.2f}GB")
        elif 'memory_gb' in data:
            print(f"  Memory used: {data['memory_gb']:.2f}GB")
    else:
        print(f"○ {model}: No test results yet")

print("\n" + "="*80)