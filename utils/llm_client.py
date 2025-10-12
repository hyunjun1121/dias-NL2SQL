"""LLM client for API calls."""

import os
from typing import Optional


class LLMClient:
    """LLM client supporting OpenAI and Anthropic."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if "gpt" in self.model_name.lower():
            return os.getenv("OPENAI_API_KEY")
        elif "claude" in self.model_name.lower():
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def _initialize_client(self):
        """Initialize API client."""
        if "gpt" in self.model_name.lower():
            import openai
            return openai.OpenAI(api_key=self.api_key)
        elif "claude" in self.model_name.lower():
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        """Generate text from prompt."""
        if "gpt" in self.model_name.lower():
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

        elif "claude" in self.model_name.lower():
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text

        raise ValueError(f"Unsupported model: {self.model_name}")
