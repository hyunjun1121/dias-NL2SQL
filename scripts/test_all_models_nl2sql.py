#!/usr/bin/env python
"""Test all three models with actual NL2SQL prompts and compare responses."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
from datetime import datetime

print("="*80)
print("NL2SQL Practical Test - All Models")
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
    },
    {
        "name": "JOIN Query",
        "schema": """
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    product_id INT,
    quantity INT,
    order_date DATE
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    price DECIMAL(10,2)
);
""",
        "question": "Get the total revenue for each product in the last month"
    },
    {
        "name": "Complex Aggregation",
        "schema": """
CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    store_id INT,
    product_id INT,
    sale_date DATE,
    amount DECIMAL(10,2)
);

CREATE TABLE stores (
    store_id INT PRIMARY KEY,
    store_name VARCHAR(100),
    region VARCHAR(50)
);
""",
        "question": "Find the top 3 stores by total sales in each region"
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

def test_model(model_name, model_path, test_prompts):
    """Test a single model with all prompts."""
    results = {
        "model": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    try:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

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
            trust_remote_code=True
        )

        load_time = time.time() - start_time
        results["load_time"] = load_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")

        # Check memory
        memory_used = 0
        for i in range(torch.cuda.device_count()):
            memory_used += torch.cuda.memory_allocated(i) / (1024**3)
        results["memory_gb"] = memory_used
        print(f"Memory used: {memory_used:.1f}GB")

        # Test each prompt
        for test_case in test_prompts:
            print(f"\n--- Test: {test_case['name']} ---")

            prompt = format_prompt_for_sql(test_case['schema'], test_case['question'])

            # Prepare input based on model type
            if "MiniMax" in model_path:
                # MiniMax format
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
            else:
                # Qwen format
                messages = [
                    {"role": "user", "content": prompt}
                ]

            # Apply chat template
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Generate
            print("Generating SQL...")
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,  # Low temperature for SQL
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                )

            gen_time = time.time() - start_time

            # Decode response
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )

            # Extract SQL from response
            sql = response
            if "```sql" in sql.lower():
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()

            # Clean up SQL
            sql = sql.strip()
            if sql.startswith("SQL Query:"):
                sql = sql[10:].strip()

            print(f"Generated SQL:\n{sql}")
            print(f"Generation time: {gen_time:.2f}s")

            # Save test result
            test_result = {
                "test_name": test_case['name'],
                "question": test_case['question'],
                "generated_sql": sql,
                "generation_time": gen_time,
                "has_thinking": "<think>" in response if "think" in model_name.lower() or "MiniMax" in model_path else False
            }
            results["tests"].append(test_result)

        # Clean up to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        results["status"] = "success"

    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results

def main():
    """Test all models and save results."""

    # Model configurations
    models = [
        {
            "name": "MiniMax-M2",
            "path": "MiniMaxAI/MiniMax-M2"
        },
        {
            "name": "Qwen3-235B-Thinking",
            "path": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
        },
        {
            "name": "Qwen3-480B-Coder",
            "path": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
        }
    ]

    all_results = {
        "test_timestamp": datetime.now().isoformat(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "models": []
    }

    # Test each model
    for model_config in models:
        print(f"\n{'='*80}")
        print(f"Preparing to test: {model_config['name']}")
        print(f"{'='*80}")

        response = input(f"Test {model_config['name']}? (y/n): ")
        if response.lower() != 'y':
            print(f"Skipping {model_config['name']}")
            continue

        results = test_model(
            model_config['name'],
            model_config['path'],
            TEST_PROMPTS
        )

        all_results["models"].append(results)

        # Save intermediate results
        os.makedirs("results", exist_ok=True)
        with open(f"results/nl2sql_test_{model_config['name'].lower().replace(' ', '_')}.json", "w") as f:
            json.dump(results, f, indent=2)

    # Save combined results
    with open("results/nl2sql_all_models_test.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    for model_result in all_results["models"]:
        if model_result["status"] == "success":
            print(f"\n{model_result['model']}:")
            print(f"  Memory: {model_result['memory_gb']:.1f}GB")
            print(f"  Load time: {model_result['load_time']:.1f}s")
            print(f"  Tests completed: {len(model_result['tests'])}")

            avg_time = sum(t['generation_time'] for t in model_result['tests']) / len(model_result['tests'])
            print(f"  Avg generation time: {avg_time:.2f}s")

    print("\n✓ All tests completed!")
    print("Results saved in results/nl2sql_all_models_test.json")

if __name__ == "__main__":
    main()