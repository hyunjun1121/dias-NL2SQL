"""
Simplified test for HuggingFace Qwen models.
Minimal test to verify model loading and basic generation.
"""

import torch
import time
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_minimal(model_type: str = "thinking"):
    """
    Minimal test to verify model loads and generates.

    Args:
        model_type: 'thinking' or 'coder'
    """

    # Model selection
    if model_type == "thinking":
        model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
        print("Testing Qwen3-235B Thinking model")
    else:
        model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
        print("Testing Qwen3-480B Coder model")

    print(f"Model: {model_name}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("-" * 60)

    # Step 1: Load tokenizer
    print("Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded in {time.time() - start_time:.2f}s")

    # Step 2: Load model
    print("Loading model...")
    start_time = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        loading_time = time.time() - start_time
        print(f"Model loaded in {loading_time:.2f}s")

    except Exception as e:
        print(f"Error loading model: {e}")
        return {
            "success": False,
            "error": str(e),
            "model": model_name
        }

    # Step 3: Test generation
    print("\nTesting generation...")

    # Simple test prompt
    test_prompt = "Generate a SQL query to count employees"

    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Input prompt: {test_prompt}")
    print(f"Tokenized length: {len(tokenizer.encode(text))} tokens")

    # Generate
    start_time = time.time()
    model_inputs = tokenizer([text], return_tensors="pt")

    if torch.cuda.is_available():
        model_inputs = model_inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True
        )

    generation_time = time.time() - start_time

    # Decode output
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Handle thinking mode
    if "Thinking" in model_name:
        try:
            # Find </think> token
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True)
            response = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
            output = f"[THINKING]: {thinking}\n[RESPONSE]: {response}"
        except ValueError:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
    else:
        output = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"\nGenerated output:")
    print("-" * 40)
    print(output[:500])  # First 500 chars
    if len(output) > 500:
        print("... (truncated)")
    print("-" * 40)

    print(f"\nGeneration time: {generation_time:.2f}s")
    print(f"Tokens generated: {len(output_ids)}")
    print(f"Tokens/second: {len(output_ids)/generation_time:.2f}")

    # GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "success": True,
        "model": model_name,
        "loading_time": loading_time,
        "generation_time": generation_time,
        "tokens_generated": len(output_ids),
        "tokens_per_second": len(output_ids)/generation_time,
        "output_sample": output[:200]
    }


def main():
    parser = argparse.ArgumentParser(description="Simple HF Qwen model test")
    parser.add_argument("--model", choices=["thinking", "coder"], default="thinking",
                      help="Model to test")
    parser.add_argument("--save_results", action="store_true",
                      help="Save results to JSON")

    args = parser.parse_args()

    print("=" * 60)
    print("Simple HuggingFace Qwen Model Test")
    print("=" * 60)
    print()

    # Run test
    results = test_model_minimal(args.model)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if results["success"]:
        print("✓ Test successful!")
        print(f"  Loading time: {results['loading_time']:.2f}s")
        print(f"  Generation time: {results['generation_time']:.2f}s")
        print(f"  Tokens/second: {results['tokens_per_second']:.2f}")
    else:
        print("✗ Test failed!")
        print(f"  Error: {results.get('error', 'Unknown')}")

    # Save results
    if args.save_results:
        filename = f"simple_test_{args.model}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()