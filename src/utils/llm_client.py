"""
Unified LLM client interface supporting multiple providers.

This module provides abstract base class and concrete implementations
for OpenAI, Anthropic, and local model LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import os


class LLMClient(ABC):
    """Abstract base class for all LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Provider-specific generation parameters

        Returns:
            Generated text string

        Raises:
            LLMClientError: If generation fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate for pricing).

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost in USD for given token counts.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            max_retries: Maximum retry attempts for failed requests
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API with retry logic."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMClientError(f"OpenAI API call failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        return ""  # Should never reach here

    def count_tokens(self, text: str) -> int:
        """Approximate token count (rough estimate: 4 chars per token)."""
        return len(text) // 4

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on OpenAI pricing."""
        # Pricing as of 2024 (approximate, adjust as needed)
        pricing = {
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
        }

        model_pricing = pricing.get(self.model, pricing["gpt-3.5-turbo"])
        prompt_cost = prompt_tokens * model_pricing["prompt"]
        completion_cost = completion_tokens * model_pricing["completion"]

        return prompt_cost + completion_cost


class AnthropicClient(LLMClient):
    """Anthropic API client with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            max_retries: Maximum retry attempts for failed requests
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API with retry logic."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMClientError(f"Anthropic API call failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        return ""  # Should never reach here

    def count_tokens(self, text: str) -> int:
        """Approximate token count (rough estimate: 4 chars per token)."""
        return len(text) // 4

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on Anthropic pricing."""
        # Pricing as of 2024 (approximate, adjust as needed)
        pricing = {
            "claude-3-opus-20240229": {"prompt": 0.015 / 1000, "completion": 0.075 / 1000},
            "claude-3-sonnet-20240229": {"prompt": 0.003 / 1000, "completion": 0.015 / 1000},
            "claude-3-haiku-20240307": {"prompt": 0.00025 / 1000, "completion": 0.00125 / 1000},
        }

        model_pricing = pricing.get(self.model, pricing["claude-3-sonnet-20240229"])
        prompt_cost = prompt_tokens * model_pricing["prompt"]
        completion_cost = completion_tokens * model_pricing["completion"]

        return prompt_cost + completion_cost


class LocalModelClient(LLMClient):
    """Local model client using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Initialize local model client.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ("cpu" or "cuda")
            max_length: Maximum generation length
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local model."""
        max_length = kwargs.get("max_length", self.max_length)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer."""
        return len(self.tokenizer.encode(text))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Local models have no API cost."""
        return 0.0


class LLMClientError(Exception):
    """Raised when LLM client operation fails."""

    pass


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    Factory function to create LLM client from configuration.

    Args:
        config: Configuration dict with keys:
            - provider: "openai", "anthropic", or "local"
            - model: Model identifier
            - temperature: Sampling temperature
            - max_tokens: Maximum completion tokens

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If provider is not supported
    """
    provider = config.get("provider")
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 512)

    if provider == "openai":
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "local":
        return LocalModelClient(
            model_name=model,
            max_length=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
