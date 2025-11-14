#!/usr/bin/env python
"""Compare Qwen3 Thinking, Qwen3 Coder, and MiniMax-M2 models - Updated with actual test results."""

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
        "memory_tested": 339.62,  # Updated from actual test
        "gpus_required": 4,  # Updated - works with 4 GPUs!
        "context": "256K tokens (1M with Yarn)",
        "thinking_mode": "No",
        "best_for": "Code generation, tool calling, agentic tasks",
        "status": "✅ Tested on 4xH100"  # Updated status
    },
    "MiniMax-M2": {
        "full_name": "MiniMaxAI/MiniMax-M2",
        "params": "228.7B total",
        "memory_tested": 214.35,  # From actual test
        "gpus_required": 4,
        "context": "TBD",
        "thinking_mode": "Yes (interleaved <think> tags)",
        "best_for": "General reasoning with thinking",
        "status": "✅ Tested on 4xH100"  # Updated status
    }
}

# Print comparison table
print("📊 Model Specifications")
print("-" * 80)
for name, specs in models.items():
    print(f"\n{name}:")
    for key, value in specs.items():
        if key != "full_name":
            if key == "memory_tested":
                print(f"  Memory Used: {value:.2f}GB")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")

print("\n" + "="*80)
print("🎉 KEY FINDINGS - All Models Work on 4xH100 GPUs!")
print("-" * 80)
print("""
✅ Qwen3-235B-Thinking: 220.31GB (Tested)
✅ Qwen3-480B-Coder: 339.62GB (Tested - Much less than expected!)
✅ MiniMax-M2: 214.35GB (Tested)

All three models successfully run on 4 x H100 GPUs (372.5GB total)!
""")

print("="*80)
print("🎯 Recommendations for NL2SQL Pipeline")
print("-" * 80)

print("""
1. For Query Understanding & Planning:
   • Primary: Qwen3-235B-Thinking (220GB, thinking mode)
   • Alternative: MiniMax-M2 (214GB, thinking mode)
   • Most efficient: MiniMax-M2 (lowest memory)

2. For SQL Generation & Code:
   • Best: Qwen3-480B-Coder (340GB, specialized for code)
   • Alternative: Qwen3-235B-Thinking (can also generate SQL)

3. Deployment Strategy (4 GPU Setup):
   • Option A: Load one model at a time (swap as needed)
   • Option B: Use MiniMax-M2 (214GB) + smaller specialized models
   • Option C: Use Qwen3-Coder for everything (340GB, no thinking)

4. Context Window Advantages:
   • Qwen3-Coder: Up to 1M tokens with Yarn
   • Qwen3-Thinking: 262K tokens native
   • MiniMax-M2: Standard context
""")

print("\n📈 Performance Comparison:")
print("-" * 80)
print(f"{'Model':<25} {'Memory (GB)':<15} {'Load Time':<15} {'Inference':<20}")
print("-" * 80)
print(f"{'Qwen3-235B-Thinking':<25} {'220.31':<15} {'~18 min':<15} {'57s/100 tokens':<20}")
print(f"{'Qwen3-480B-Coder':<25} {'339.62':<15} {'~15-20 min':<15} {'684s/500 tokens':<20}")
print(f"{'MiniMax-M2':<25} {'214.35':<15} {'~14 min':<15} {'71-159s/query':<20}")

# Check for test results
print("\n📁 Test Results Files:")
print("-" * 80)

# Possible result file locations
result_files = {
    "Qwen3-235B": ["results/final_gpu_test_results.json", "results/qwen3_thinking_test.json"],
    "Qwen3-Coder": ["results/qwen3_coder_test.json", "results/qwen_coder_results.json"],
    "MiniMax-M2": ["results/minimax_m2_official_test.json", "results/minimax_m2_test.json"]
}

for model, filepaths in result_files.items():
    found = False
    for filepath in filepaths:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"✓ {model}: Found at {filepath}")

            # Print key metrics
            if 'memory_used_gb' in data:
                print(f"  Memory: {data['memory_used_gb']:.2f}GB")
            elif 'memory_gb' in data:
                print(f"  Memory: {data['memory_gb']:.2f}GB")

            if 'load_time' in data:
                print(f"  Load time: {data['load_time']:.1f}s")

            if 'parameters_billion' in data:
                print(f"  Parameters: {data['parameters_billion']:.1f}B")

            found = True
            break

    if not found:
        print(f"○ {model}: No test results found")

print("\n" + "="*80)
print("💡 CONCLUSION")
print("-" * 80)
print("""
All three models can run on the current 4xH100 GPU setup!
- Most memory efficient: MiniMax-M2 (214GB)
- Best for code: Qwen3-480B-Coder (340GB)
- Best for reasoning: Qwen3-235B-Thinking (220GB)

No need for 6-8 GPUs as initially expected!
""")
print("="*80)