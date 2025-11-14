"""
Test script for Qwen3 large models with HuggingFace Transformers.
Tests memory requirements and performance on H100 GPUs.
"""

import os
import time
import torch
import psutil
import gc
from typing import Optional, Dict, Any
from dataclasses import dataclass
import argparse
import json
from datetime import datetime

# GPU memory monitoring
def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_reserved_gb': max_reserved,
            'device_count': torch.cuda.device_count()
        }
    return None

def get_system_memory_info():
    """Get system RAM usage."""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1024**3,
        'available_gb': mem.available / 1024**3,
        'used_gb': mem.used / 1024**3,
        'percent': mem.percent
    }

@dataclass
class ModelTestResult:
    """Results from model testing."""
    model_name: str
    loading_time: float
    inference_time: float
    tokens_per_second: float
    gpu_memory_used: Dict
    system_memory_used: Dict
    num_gpus: int
    success: bool
    error: Optional[str] = None
    output_sample: Optional[str] = None

class QwenModelTester:
    """Test Qwen models with HuggingFace."""

    def __init__(self, model_name: str, num_gpus: Optional[int] = None):
        self.model_name = model_name
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model with optimal settings."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")
        print(f"Available GPUs: {torch.cuda.device_count()}")

        start_time = time.time()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configure device map for multi-GPU
            if self.num_gpus > 1:
                device_map = "auto"
                print(f"Using automatic device mapping across {self.num_gpus} GPUs")
            else:
                device_map = {"": 0}
                print("Using single GPU")

            # Load model with FP8 quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",  # Will use FP8 from config
                device_map=device_map,
                trust_remote_code=True,
                # Additional optimizations
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            loading_time = time.time() - start_time
            print(f"Model loaded in {loading_time:.2f} seconds")

            return loading_time

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def test_inference(self, prompt: str, max_new_tokens: int = 512):
        """Test model inference."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        print(f"Testing inference with prompt: '{prompt[:100]}...'")

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(self.model.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )

        inference_time = time.time() - start_time

        # Decode output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Handle thinking mode for Qwen3-235B
        if "Thinking" in self.model_name:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True)
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
                output = f"[THINKING]\n{thinking_content}\n[RESPONSE]\n{content}"
            except ValueError:
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Calculate tokens per second
        num_tokens = len(output_ids)
        tokens_per_second = num_tokens / inference_time if inference_time > 0 else 0

        print(f"Generated {num_tokens} tokens in {inference_time:.2f}s ({tokens_per_second:.2f} tokens/s)")

        return output, inference_time, tokens_per_second

    def run_test(self, test_prompt: str) -> ModelTestResult:
        """Run complete test."""
        result = ModelTestResult(
            model_name=self.model_name,
            loading_time=0,
            inference_time=0,
            tokens_per_second=0,
            gpu_memory_used={},
            system_memory_used={},
            num_gpus=self.num_gpus,
            success=False
        )

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Record initial memory
            initial_gpu_mem = get_gpu_memory_info()
            initial_sys_mem = get_system_memory_info()

            # Load model
            loading_time = self.load_model()
            result.loading_time = loading_time

            # Record memory after loading
            loaded_gpu_mem = get_gpu_memory_info()
            loaded_sys_mem = get_system_memory_info()

            # Test inference
            output, inference_time, tps = self.test_inference(test_prompt)
            result.inference_time = inference_time
            result.tokens_per_second = tps
            result.output_sample = output[:500]  # First 500 chars

            # Final memory usage
            final_gpu_mem = get_gpu_memory_info()
            final_sys_mem = get_system_memory_info()

            result.gpu_memory_used = {
                'initial': initial_gpu_mem,
                'after_loading': loaded_gpu_mem,
                'after_inference': final_gpu_mem
            }
            result.system_memory_used = {
                'initial': initial_sys_mem,
                'after_loading': loaded_sys_mem,
                'after_inference': final_sys_mem
            }

            result.success = True
            print("Test completed successfully!")

        except Exception as e:
            result.error = str(e)
            print(f"Test failed: {e}")

        finally:
            # Cleanup
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return result

def estimate_gpu_requirements(model_name: str) -> Dict:
    """Estimate GPU requirements for model."""
    estimates = {
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": {
            "min_gpus": 4,
            "recommended_gpus": 8,
            "memory_per_gpu_gb": 80,
            "total_memory_gb": 235,
            "notes": "FP8 quantized, MoE with 22B active params"
        },
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
            "min_gpus": 8,
            "recommended_gpus": 16,
            "memory_per_gpu_gb": 80,
            "total_memory_gb": 480,
            "notes": "FP8 quantized, MoE with 35B active params"
        }
    }

    return estimates.get(model_name, {
        "min_gpus": "Unknown",
        "notes": "Model not in database"
    })

def main():
    parser = argparse.ArgumentParser(description="Test Qwen large models")
    parser.add_argument("--model", type=str, required=True,
                      choices=["thinking", "coder"],
                      help="Which model to test (thinking or coder)")
    parser.add_argument("--num_gpus", type=int, default=None,
                      help="Number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=512,
                      help="Maximum tokens to generate")
    parser.add_argument("--prompt", type=str,
                      default="Explain the concept of attention mechanism in transformers.",
                      help="Test prompt")
    parser.add_argument("--output_file", type=str, default=None,
                      help="JSON file to save results")

    args = parser.parse_args()

    # Select model
    if args.model == "thinking":
        model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
    else:
        model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}\n")

    # Show estimates
    estimates = estimate_gpu_requirements(model_name)
    print("GPU Requirements Estimate:")
    for key, value in estimates.items():
        print(f"  {key}: {value}")
    print()

    # Run test
    tester = QwenModelTester(model_name, args.num_gpus)
    result = tester.run_test(args.prompt)

    # Display results
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Model: {result.model_name}")
    print(f"Success: {result.success}")
    print(f"Loading Time: {result.loading_time:.2f}s")
    print(f"Inference Time: {result.inference_time:.2f}s")
    print(f"Tokens/Second: {result.tokens_per_second:.2f}")
    print(f"GPUs Used: {result.num_gpus}")

    if result.error:
        print(f"Error: {result.error}")

    if result.gpu_memory_used:
        final_gpu = result.gpu_memory_used.get('after_inference', {})
        if final_gpu:
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {final_gpu.get('allocated_gb', 0):.2f} GB")
            print(f"  Reserved: {final_gpu.get('reserved_gb', 0):.2f} GB")
            print(f"  Max Reserved: {final_gpu.get('max_reserved_gb', 0):.2f} GB")

    if result.output_sample:
        print(f"\nOutput Sample:\n{result.output_sample}")

    # Save results
    if args.output_file:
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'model_name': result.model_name,
            'success': result.success,
            'loading_time': result.loading_time,
            'inference_time': result.inference_time,
            'tokens_per_second': result.tokens_per_second,
            'num_gpus': result.num_gpus,
            'error': result.error,
            'gpu_memory': result.gpu_memory_used,
            'system_memory': result.system_memory_used,
            'output_sample': result.output_sample
        }

        with open(args.output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()