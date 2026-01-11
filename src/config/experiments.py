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
    api_key_env_var: Optional[str] = Field(
        None, description="Environment variable name for API key"
    )

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
    k1: float = Field(1.5, ge=0.0, description="BM25 term frequency saturation")
    b: float = Field(0.75, ge=0.0, le=1.0, description="BM25 length normalization")
    embedding_model: Optional[str] = Field(
        "all-MiniLM-L6-v2", description="For dense/hybrid retrieval"
    )
    reranking: bool = Field(False, description="Whether to apply reranking")
    bm25_weight: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Weight for BM25 in hybrid mode"
    )
    dense_weight: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Weight for dense in hybrid mode"
    )

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
    description: str = Field(
        ..., min_length=1, description="What this experiment tests"
    )
    run_id: str = Field(..., min_length=1, description="Unique identifier for this run")
    pipeline_type: str = Field(..., description="baseline, multiagent, or specialized")
    retrieval_config: RetrievalConfig = Field(
        ..., description="Retrieval strategy configuration"
    )
    llm_configs: Dict[str, LLMConfig] = Field(
        ..., description="LLM config per agent or single for baseline"
    )
    agent_architecture: Optional[List[str]] = Field(
        None, description="Agent types in execution order (multiagent only)"
    )
    orchestration_mode: Optional[str] = Field(
        None, description="sequential or parallel (multiagent only)"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional hyperparameters"
    )

    @field_validator("pipeline_type")
    @classmethod
    def validate_pipeline_type(cls, v: str) -> str:
        """Validate pipeline type is supported."""
        # Only baseline is currently implemented
        # Future: Add support for multiagent and specialized pipelines
        valid_types = {"baseline"}
        if v not in valid_types:
            raise ValueError(f"pipeline_type must be 'baseline' (only supported type currently)")
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
        """Validate pipeline-specific requirements."""
        # Multi-agent validation removed - not currently implemented
        pass

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


class LangfuseExperimentConfig(ExperimentConfig):
    """
    Extended experiment configuration with Langfuse-specific settings.

    Inherits all standard experiment config fields and adds:
    - Dataset management (Langfuse or local)
    - Tracing configuration
    - Concurrency settings
    - Evaluator configuration
    - Tags and metadata propagation
    """

    # Dataset Configuration
    langfuse_dataset_name: Optional[str] = Field(
        default=None, description="Langfuse dataset name to run experiment against"
    )

    use_local_data: bool = Field(
        default=False,
        description="Whether to use local FinDER data instead of Langfuse dataset",
    )

    local_data_path: Optional[Path] = Field(
        default=None, description="Path to local data if use_local_data=True"
    )

    # Langfuse Tracing Configuration
    flush_interval: float = Field(
        default=5.0, gt=0.0, description="Seconds between automatic trace flushes"
    )

    flush_batch_size: int = Field(
        default=100, ge=1, description="Number of events before automatic flush"
    )

    enable_tracing: bool = Field(
        default=True,
        description="Whether to enable Langfuse tracing (can be overridden by env var)",
    )

    # Concurrency Configuration
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Maximum concurrent task executions (1 = sequential)",
    )

    # Evaluator Configuration
    enable_item_evaluators: bool = Field(
        default=True, description="Whether to run item-level evaluators"
    )

    enable_run_evaluators: bool = Field(
        default=True, description="Whether to run run-level (aggregate) evaluators"
    )

    item_evaluator_names: List[str] = Field(
        default_factory=lambda: ["token_f1"],
        description="Names of item-level evaluators to run",
    )

    run_evaluator_names: List[str] = Field(
        default_factory=lambda: ["average_accuracy"],
        description="Names of run-level evaluators to run",
    )

    evaluation_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Minimum acceptable values for evaluators (e.g., {'token_f1': 0.5})",
    )

    # Metadata and Tags
    langfuse_tags: List[str] = Field(
        default_factory=list,
        description="Tags to attach to all traces in this experiment",
    )

    langfuse_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata to attach to all traces in this experiment",
    )

    propagate_query_metadata: bool = Field(
        default=True, description="Whether to propagate FinDER query metadata to traces"
    )

    @field_validator("item_evaluator_names")
    @classmethod
    def validate_item_evaluators(cls, v: List[str]) -> List[str]:
        """Validate evaluator names are recognized."""
        valid_evaluators = {
            "token_f1",
            "exact_match",
            "semantic_similarity",
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_quality",
            "answer_correctness",
        }

        for evaluator in v:
            if evaluator not in valid_evaluators:
                raise ValueError(
                    f"Unknown item evaluator: {evaluator}. "
                    f"Valid options: {valid_evaluators}"
                )

        return v

    @field_validator("run_evaluator_names")
    @classmethod
    def validate_run_evaluators(cls, v: List[str]) -> List[str]:
        """Validate run-level evaluator names are recognized."""
        valid_evaluators = {
            "average_accuracy",
            "aggregate_retrieval_metrics",
            "pass_rate",
            "regression_check",
        }

        for evaluator in v:
            if evaluator not in valid_evaluators:
                raise ValueError(
                    f"Unknown run evaluator: {evaluator}. "
                    f"Valid options: {valid_evaluators}"
                )

        return v
