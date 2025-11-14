"""
HuggingFace Transformers LLM Client for Qwen large models.
Alternative to Ollama for running models directly with transformers.
"""

import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc

class HFLLMClient:
    """
    LLM Client using HuggingFace Transformers directly.
    Supports Qwen3-235B and Qwen3-Coder-480B models.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        max_memory: Optional[Dict] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize HuggingFace LLM client.

        Args:
            model_name: HuggingFace model identifier
            device_map: Device mapping strategy ('auto', 'balanced', etc.)
            max_memory: Maximum memory per device
            load_in_8bit: Use 8-bit quantization
        """
        self.model_name = model_name
        self.device_map = device_map
        self.max_memory = max_memory
        self.load_in_8bit = load_in_8bit

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

        # Model type detection
        self.is_thinking_model = "Thinking" in model_name
        self.is_coder_model = "Coder" in model_name

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Model loading arguments
        model_kwargs = {
            "torch_dtype": "auto",  # Will use FP8 if specified in config
            "device_map": self.device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Add flash attention if available
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Add max memory if specified
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory

        # Add 8-bit quantization if requested
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        print(f"Model loaded successfully")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 20,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                **kwargs
            )

        # Decode output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Handle thinking mode for Qwen3-235B
        if self.is_thinking_model:
            return self._parse_thinking_output(output_ids)
        else:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _parse_thinking_output(self, output_ids: List[int]) -> str:
        """
        Parse thinking model output.

        Args:
            output_ids: Generated token IDs

        Returns:
            Parsed output with thinking and response separated
        """
        try:
            # Find </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True)
            response_content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

            # For NL2SQL, we mainly need the response
            # But include thinking for debugging
            return response_content.strip()

        except ValueError:
            # No thinking tags found
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []

        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            results.append(result)

        return results

    def cleanup(self):
        """Clean up model and free memory."""
        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class HFLLMClientAdapter:
    """
    Adapter to make HFLLMClient compatible with existing LLMClient interface.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize adapter.

        Args:
            model_name: Model identifier (e.g., "hf:qwen3-235b-thinking")
            **kwargs: Additional arguments for HFLLMClient
        """
        # Parse model name
        if model_name.startswith("hf:"):
            model_name = model_name[3:]

        # Map short names to full HuggingFace model IDs
        model_mapping = {
            "qwen3-235b-thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
            "qwen3-480b-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        }

        if model_name in model_mapping:
            full_model_name = model_mapping[model_name]
        else:
            full_model_name = model_name

        # Initialize underlying client
        self.client = HFLLMClient(full_model_name, **kwargs)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, **kwargs) -> str:
        """Generate text (compatible with LLMClient interface)."""
        return self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def cleanup(self):
        """Clean up resources."""
        self.client.cleanup()