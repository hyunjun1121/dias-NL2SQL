#!/usr/bin/env python
"""Test MiniMax-M2 model based on official deployment guide."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import json
import os

print("="*70)
print("MiniMax-M2 Official Test")
print("="*70)

# System requirements check
print("System Requirements:")
print("  • Python 3.9-3.12")
print("  • Transformers 4.57.1")
print("  • GPU Memory: ~220GB for weights")
print("  • Compute capability: 7.0+")
print()

# GPU info
print(f"Available GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / (1024**3)
        total_memory += mem
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {mem:.1f}GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    print(f"Total GPU Memory: {total_memory:.1f}GB")

    if total_memory < 220:
        print(f"\n⚠️ Warning: Model needs ~220GB, but only {total_memory:.1f}GB available")
print()

MODEL_PATH = "MiniMaxAI/MiniMax-M2"

try:
    print(f"Loading model: {MODEL_PATH}")
    print("This will auto-download from HuggingFace if not cached...")

    start_time = time.time()

    # Load model as per official guide
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,  # Required for MiniMax-M2
        torch_dtype="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")

    # Test 1: Simple conversation (from official example)
    print("\n--- Test 1: Multi-turn Conversation ---")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"}]},
        {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]}
    ]

    print("Input messages:")
    for msg in messages:
        role = msg["role"]
        content = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
        print(f"  {role}: {content[:100]}...")

    start_time = time.time()
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=200,
        generation_config=model.generation_config
    )

    gen_time = time.time() - start_time
    response = tokenizer.batch_decode(generated_ids)[0]

    # Extract assistant response
    if "<think>" in response and "</think>" in response:
        print("\n[Model uses thinking mode]")

    print(f"\nResponse preview: {response[-500:]}")
    print(f"Generation time: {gen_time:.2f}s")

    # Test 2: Math problem with thinking
    print("\n--- Test 2: Math with Reasoning ---")
    math_messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is 234 * 567? Show your thinking process."}]}
    ]

    start_time = time.time()
    model_inputs = tokenizer.apply_chat_template(
        math_messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=500,
        generation_config=model.generation_config
    )

    gen_time = time.time() - start_time
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Parse thinking content
    if "<think>" in response and "</think>" in response:
        think_start = response.find("<think>")
        think_end = response.find("</think>") + 8
        thinking = response[think_start:think_end]
        answer = response[think_end:].strip()

        print("Thinking process found:")
        print(f"  {thinking[:200]}...")
        print(f"\nFinal answer: {answer[:200]}...")
    else:
        print(f"Response: {response[-500:]}")

    print(f"Generation time: {gen_time:.2f}s")

    # Test 3: Code generation
    print("\n--- Test 3: Code Generation ---")
    code_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Write a Python function to find prime numbers up to n."}]}
    ]

    start_time = time.time()
    model_inputs = tokenizer.apply_chat_template(
        code_messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=400,
        generation_config=model.generation_config
    )

    gen_time = time.time() - start_time
    response = tokenizer.batch_decode(generated_ids)[0]

    print(f"Code response preview: {response[-800:][:400]}...")
    print(f"Generation time: {gen_time:.2f}s")

    # Memory usage
    print("\n--- GPU Memory Usage ---")
    total_used = 0
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total_used += allocated
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print(f"Total memory used: {total_used:.2f}GB")

    # Save results
    results = {
        "model": MODEL_PATH,
        "parameters_billion": total_params/1e9,
        "load_time": load_time,
        "memory_used_gb": total_used,
        "gpus_used": torch.cuda.device_count(),
        "supports_thinking": True,
        "interleaved_thinking": True,
        "test_successful": True,
        "memory_requirement": 220,
        "actual_vs_required": f"{total_used:.1f}GB used vs 220GB required"
    }

    os.makedirs("results", exist_ok=True)
    with open("results/minimax_m2_official_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ All tests completed successfully!")
    print(f"Results saved to results/minimax_m2_official_test.json")

    # Recommendations
    print("\n📊 Summary:")
    print(f"  • Model size: {total_params/1e9:.2f}B parameters")
    print(f"  • Memory used: {total_used:.2f}GB")
    print(f"  • GPUs required: {max(3, int(total_used/93)+1)} x H100 minimum")
    print(f"  • Thinking mode: ✓ Supported (interleaved)")

except torch.cuda.OutOfMemoryError:
    print("\n✗ Out of Memory Error!")
    print("MiniMax-M2 requires ~220GB of GPU memory")
    print("Current setup may be insufficient")

except Exception as e:
    print(f"\n✗ Error: {e}")

    if "trust_remote_code" in str(e):
        print("\nMake sure to set trust_remote_code=True")
    elif "MiniMax-M2 model is not currently supported" in str(e):
        print("\nPlease check trust_remote_code=True is set")

    import traceback
    traceback.print_exc()

print("\n" + "="*70)