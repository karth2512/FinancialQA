"""
Experiment configuration management with Pydantic validation.

This module defines type-safe configuration schemas for experiments,
retrieval strategies, and LLM providers.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: str = Field(..., description="openai, anthropic, or local")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(512, ge=1, description="Max completion tokens")
    api_key_env_var: Optional[str] = Field(None, description="Environment variable name for API key")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        valid_providers = {"openai", "anthropic", "local"}
        if v not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}")
        return v


class RetrievalConfig(BaseModel):
    """Configuration for retrieval strategy."""

    strategy: str = Field(..., description="bm25, dense, or hybrid")
    top_k: int = Field(5, ge=1, description="Number of passages to retrieve")
    embedding_model: Optional[str] = Field("all-MiniLM-L6-v2", description="For dense/hybrid retrieval")
    reranking: bool = Field(False, description="Whether to apply reranking")
    bm25_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Weight for BM25 in hybrid mode")
    dense_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Weight for dense in hybrid mode")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate retrieval strategy is supported."""
        valid_strategies = {"bm25", "dense", "hybrid"}
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for an evaluation experiment."""

    name: str = Field(..., min_length=1, description="Experiment name")
    description: str = Field(..., min_length=1, description="What this experiment tests")
    run_id: str = Field(..., min_length=1, description="Unique identifier for this run")
    pipeline_type: str = Field(..., description="baseline, multiagent, or specialized")
    retrieval_config: RetrievalConfig = Field(..., description="Retrieval strategy configuration")
    llm_configs: Dict[str, LLMConfig] = Field(..., description="LLM config per agent or single for baseline")
    agent_architecture: Optional[List[str]] = Field(None, description="Agent types in execution order (multiagent only)")
    orchestration_mode: Optional[str] = Field(None, description="sequential or parallel (multiagent only)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Additional hyperparameters")

    @field_validator("pipeline_type")
    @classmethod
    def validate_pipeline_type(cls, v: str) -> str:
        """Validate pipeline type is supported."""
        valid_types = {"baseline", "multiagent", "specialized"}
        if v not in valid_types:
            raise ValueError(f"pipeline_type must be one of {valid_types}")
        return v

    @field_validator("orchestration_mode")
    @classmethod
    def validate_orchestration_mode(cls, v: Optional[str]) -> Optional[str]:
        """Validate orchestration mode if provided."""
        if v is not None:
            valid_modes = {"sequential", "parallel"}
            if v not in valid_modes:
                raise ValueError(f"orchestration_mode must be one of {valid_modes}")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate multiagent requirements."""
        if self.pipeline_type == "multiagent":
            if not self.agent_architecture:
                raise ValueError("agent_architecture must be non-empty for multiagent pipeline")

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """
        Load experiment configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated ExperimentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """
        Save experiment configuration to YAML file.

        Args:
            path: Path to save YAML configuration file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
