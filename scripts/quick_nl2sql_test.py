#!/usr/bin/env python
"""Quick test - send same SQL prompt to all loaded models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

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

def quick_test_model(model_name, model_path):
    """Quick test for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        # Load model
        print("Loading model...")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        load_time = time.time() - start
        print(f"Loaded in {load_time:.1f}s")

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

        # Memory check
        mem = sum(torch.cuda.memory_allocated(i) / 1024**3 for i in range(torch.cuda.device_count()))
        print(f"Memory used: {mem:.1f}GB")

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

        return sql.strip(), gen_time

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    print("="*80)
    print("Quick NL2SQL Test - Same Query for All Models")
    print("="*80)
    print(f"\nQuestion: {QUESTION}\n")

    models = [
        ("MiniMax-M2", "MiniMaxAI/MiniMax-M2"),
        ("Qwen3-235B-Thinking", "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"),
        ("Qwen3-480B-Coder", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
    ]

    results = []

    for name, path in models:
        choice = input(f"\nTest {name}? (y/n): ")
        if choice.lower() == 'y':
            sql, time_taken = quick_test_model(name, path)
            if sql:
                results.append((name, sql, time_taken))

    # Compare results
    if results:
        print("\n" + "="*80)
        print("COMPARISON OF GENERATED SQL")
        print("="*80)

        for name, sql, time_taken in results:
            print(f"\n{name} ({time_taken:.1f}s):")
            print("-" * 40)
            print(sql)

if __name__ == "__main__":
    main()