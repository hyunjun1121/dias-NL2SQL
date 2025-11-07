"""
LLM client for API calls.

Supports:
- Proprietary: OpenAI (GPT-4o), Anthropic (Claude)
- Open-source: vLLM cluster, Transformers, Ollama
- Custom: Cluster endpoints
"""

import os
from typing import Optional


class LLMClient:
    """
    Unified LLM client.

    Supported backends:
    - 'gpt-4o', 'gpt-4o-mini': OpenAI API
    - 'claude-3.5-sonnet': Anthropic API
    - 'qwen3-* (via HF inference)', 'deepseek-r1', 'llama-3.3': vLLM or HF
    - 'hf:{model_path}': HuggingFace Transformers (local)
    - 'ollama:{model}': Ollama local
    - 'cluster:{endpoint}': Custom cluster endpoint
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None  # For cluster endpoints
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.backend = self._detect_backend()
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()

    def _detect_backend(self) -> str:
        """Detect which backend to use."""
        model_lower = self.model_name.lower()

        # Explicit backends must be detected first to avoid false matches (e.g., 'ollama:qwen*')
        if model_lower.startswith("ollama:"):
            return "ollama"
        if model_lower.startswith("hf:"):
            return "transformers"

        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif model_lower.startswith("qwen/qwen3-") or model_lower in {
            "qwen3-235b-a22b-thinking-2507-fp8",
            "qwen3-235b-a22b-instruct-2507-fp8",
            "qwen3-coder-480b-a35b-instruct-fp8"
        }:
            return "hf_inference"
        elif any(x in model_lower for x in ["deepseek", "qwen", "llama", "mistral"]):
            return "vllm"  # vLLM cluster
        elif model_lower.startswith("cluster:"):
            return "cluster"
        else:
            return "vllm"  # Default to vLLM for unknown open-source models

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.backend == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.backend == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def _initialize_client(self):
        """Initialize client based on backend."""
        if self.backend == "openai":
            import openai
            return openai.OpenAI(api_key=self.api_key)

        elif self.backend == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)

        elif self.backend == "vllm":
            # vLLM cluster - uses OpenAI-compatible API
            import openai
            base_url = self.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            return openai.OpenAI(api_key="EMPTY", base_url=base_url)

        elif self.backend == "hf_inference":
            from huggingface_hub import InferenceClient

            token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            return InferenceClient(model=self.model_name, token=token)

        elif self.backend == "transformers":
            # HuggingFace Transformers - load model locally
            return self._load_transformers_model()

        elif self.backend == "ollama":
            # Ollama - local deployment
            return self._initialize_ollama()

        elif self.backend == "cluster":
            # Custom cluster endpoint
            import openai
            return openai.OpenAI(api_key="EMPTY", base_url=self.base_url)

        return None

    def _load_transformers_model(self):
        """Load HuggingFace model locally."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = self.model_name.replace("hf:", "")
            print(f"Loading model from HuggingFace: {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            return {"model": model, "tokenizer": tokenizer}
        except ImportError:
            raise ImportError("transformers and torch required for HuggingFace models")

    def _initialize_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            return ollama
        except ImportError:
            raise ImportError("ollama-python required: pip install ollama")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        """Generate text from prompt."""

        if self.backend == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text

        elif self.backend == "vllm" or self.backend == "cluster":
            # vLLM uses OpenAI-compatible API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.backend == "hf_inference":
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self.client.text_generation(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.05
            )
            return response.strip()

        elif self.backend == "transformers":
            # HuggingFace Transformers local generation
            model = self.client["model"]
            tokenizer = self.client["tokenizer"]

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response[len(full_prompt):].strip()
            return response

        elif self.backend == "ollama":
            # Ollama local generation
            model_name = self.model_name.replace("ollama:", "")
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            response = self.client.generate(
                model=model_name,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            # Handle both regular response and thinking mode
            response_text = response.get('response', '') if isinstance(response, dict) else getattr(response, 'response', '')
            thinking_text = response.get('thinking', '') if isinstance(response, dict) else getattr(response, 'thinking', '')

            # Prefer response field, use thinking only if response is truly empty
            if response_text:
                return response_text
            elif thinking_text:
                return thinking_text
            return ""

        raise ValueError(f"Unsupported backend: {self.backend}")
