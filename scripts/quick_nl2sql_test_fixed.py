#!/usr/bin/env python
"""Quick test - send same SQL prompt to all loaded models - Fixed version."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import json
import os
from datetime import datetime

# Simple test case
SCHEMA = """
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE
);
"""

QUESTION = "Find the average salary for each department, showing only departments with more than 5 employees"

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    gc.collect()

def quick_test_model(model_name, model_path):
    """Quick test for a single model with better cleanup."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    result = {"model": model_name}

    try:
        # Clear memory before loading
        clear_gpu_memory()
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial memory: {initial_mem:.2f}GB")

        # Load model
        print("Loading model...")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Important for memory efficiency
        )

        load_time = time.time() - start
        print(f"Loaded in {load_time:.1f}s")

        # Check memory after loading
        loaded_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after loading: {loaded_mem:.2f}GB")

        # Format prompt
        prompt = f"""Given the database schema below, write a SQL query.

Schema:
{SCHEMA}

Question: {QUESTION}

SQL Query:"""

        # Prepare messages based on model
        if "MiniMax" in model_path:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Generate
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print("\nGenerating SQL...")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
            )

        gen_time = time.time() - start

        # Decode
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # Extract SQL
        sql = response
        if "```" in sql:
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1]
                if sql.startswith("sql"):
                    sql = sql[3:]

        print(f"\nGenerated SQL:")
        print("-" * 40)
        print(sql.strip())
        print("-" * 40)
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Memory used: {loaded_mem:.1f}GB")

        result["sql"] = sql.strip()
        result["load_time"] = load_time
        result["generation_time"] = gen_time
        result["memory_gb"] = loaded_mem
        result["status"] = "success"

    except Exception as e:
        print(f"Error: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    finally:
        # Aggressive cleanup
        print("\nCleaning up...")

        # Delete variables
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs

        # Clear memory multiple times
        for _ in range(3):
            clear_gpu_memory()
            time.sleep(1)

        final_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after cleanup: {final_mem:.2f}GB")

    return result

def main():
    print("="*80)
    print("Quick NL2SQL Test - Same Query for All Models (Fixed)")
    print("="*80)
    print(f"\nQuestion: {QUESTION}\n")

    # Check GPU status
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Models to test (ordered by size)
    models = [
        ("MiniMax-M2", "MiniMaxAI/MiniMax-M2"),
        ("Qwen3-235B-Thinking", "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"),
        ("Qwen3-480B-Coder", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
    ]

    results = []

    print("\n" + "="*80)
    print("Testing all models automatically...")
    print("="*80)

    for name, path in models:
        # Clear memory before each model
        clear_gpu_memory()
        print(f"\nPreparing for: {name}")
        print(f"Current memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # Test model
        result = quick_test_model(name, path)
        results.append(result)

        # Wait between models
        print("\nWaiting 5 seconds before next model...")
        time.sleep(5)
        clear_gpu_memory()

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON OF GENERATED SQL")
    print("="*80)

    for result in results:
        if result["status"] == "success":
            print(f"\n{result['model']}:")
            print(f"  Load: {result['load_time']:.1f}s, Generate: {result['generation_time']:.1f}s, Memory: {result['memory_gb']:.1f}GB")
            print("-" * 40)
            print(result['sql'])
        else:
            print(f"\n{result['model']}: ERROR - {result.get('error', 'Unknown error')}")

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "schema": SCHEMA,
        "results": results
    }

    with open("results/quick_nl2sql_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✓ Results saved to results/quick_nl2sql_comparison.json")

if __name__ == "__main__":
    main()