#!/usr/bin/env python
"""Test all three models with actual NL2SQL prompts - Fixed memory management."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
import gc
from datetime import datetime

print("="*80)
print("NL2SQL Practical Test - All Models (Fixed)")
print("="*80)

# Test prompts for NL2SQL
TEST_PROMPTS = [
    {
        "name": "Simple SELECT",
        "schema": """
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    city VARCHAR(50)
);
""",
        "question": "Find all users who are older than 25 and live in Seoul"
    }
]

def format_prompt_for_sql(schema, question):
    """Format prompt for SQL generation."""
    return f"""Given the following database schema, write a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}

Please provide only the SQL query without any explanation.
SQL Query:"""

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

def test_single_model(model_name, model_path):
    """Test a single model and immediately clean up."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Clear memory before loading
        clear_gpu_memory()
        print("Cleared GPU memory before loading")

        # Check initial memory
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_mem:.2f}GB")

        # Load model
        print(f"Loading {model_name}...")
        start_time = time.time()

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

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")

        # Check memory after loading
        loaded_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after loading: {loaded_mem:.2f}GB")
        result["memory_gb"] = loaded_mem
        result["load_time"] = load_time

        # Test with single prompt
        test_case = TEST_PROMPTS[0]
        prompt = format_prompt_for_sql(test_case['schema'], test_case['question'])

        # Prepare input
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

        print("Generating SQL...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
            )

        gen_time = time.time() - start_time

        # Decode
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # Extract SQL
        sql = response
        if "```sql" in sql.lower():
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            parts = sql.split("```")
            if len(parts) >= 2:
                sql = parts[1].strip()

        print(f"Generated SQL:\n{sql}")
        print(f"Generation time: {gen_time:.2f}s")

        result["sql"] = sql
        result["generation_time"] = gen_time
        result["status"] = "success"

    except Exception as e:
        print(f"Error: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    finally:
        # Aggressive cleanup
        print("\nCleaning up...")

        # Delete model and tokenizer
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs

        # Clear GPU memory multiple times
        for _ in range(3):
            clear_gpu_memory()
            time.sleep(1)

        # Final memory check
        final_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after cleanup: {final_mem:.2f}GB")

        # Extra aggressive cleanup
        gc.collect()

    return result

def main():
    """Test models sequentially with aggressive memory management."""

    # Check GPU status
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Models to test (order by size - smallest first)
    models = [
        ("MiniMax-M2", "MiniMaxAI/MiniMax-M2"),
        ("Qwen3-235B-Thinking", "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"),
        ("Qwen3-480B-Coder", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
    ]

    all_results = []

    print("\n" + "="*80)
    print("Testing all models sequentially...")
    print("="*80)

    for model_name, model_path in models:
        # Clear memory before each model
        clear_gpu_memory()
        print(f"\n{'='*80}")
        print(f"Preparing for: {model_name}")
        print(f"Current memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"{'='*80}")

        # Test model
        result = test_single_model(model_name, model_path)
        all_results.append(result)

        # Save individual result
        os.makedirs("results", exist_ok=True)
        with open(f"results/nl2sql_{model_name.lower().replace(' ', '_').replace('-', '_')}.json", "w") as f:
            json.dump(result, f, indent=2)

        # Wait between models
        print("\nWaiting 5 seconds before next model...")
        time.sleep(5)
        clear_gpu_memory()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    for result in all_results:
        print(f"\n{result['model']}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Memory: {result.get('memory_gb', 0):.1f}GB")
            print(f"  Load time: {result.get('load_time', 0):.1f}s")
            print(f"  Generation time: {result.get('generation_time', 0):.2f}s")
            print(f"  SQL: {result.get('sql', 'N/A')[:100]}...")

    # Save combined results
    with open("results/nl2sql_all_sequential.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✓ All tests completed!")
    print("Results saved in results/")

if __name__ == "__main__":
    main()