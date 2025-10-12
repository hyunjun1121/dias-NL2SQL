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
    - 'deepseek-r1', 'qwen2.5', 'llama-3.3': vLLM cluster
    - 'hf:{model_path}': HuggingFace Transformers
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
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url
        self.backend = self._detect_backend()
        self.client = self._initialize_client()

    def _detect_backend(self) -> str:
        """Detect which backend to use."""
        model_lower = self.model_name.lower()

        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif any(x in model_lower for x in ["deepseek", "qwen", "llama", "mistral"]):
            return "vllm"  # vLLM cluster
        elif model_lower.startswith("hf:"):
            return "transformers"
        elif model_lower.startswith("ollama:"):
            return "ollama"
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
            return response['response']

        raise ValueError(f"Unsupported backend: {self.backend}")
