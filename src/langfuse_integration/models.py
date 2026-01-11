"""
Data models for Langfuse integration.

This module defines Pydantic models for configuring datasets, experiments,
evaluations, and managing data transformations for Langfuse SDK integration.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetFilterType(str, Enum):
    """Supported dataset filtering criteria."""

    ALL = "all"
    REASONING_REQUIRED = "reasoning_required"
    NO_REASONING = "no_reasoning"
    BY_CATEGORY = "by_category"
    BY_QUERY_TYPE = "by_query_type"
    BY_COMPLEXITY = "by_complexity"


class LangfuseDatasetConfig(BaseModel):
    """
    Configuration for uploading FinDER dataset to Langfuse.

    This model defines how the FinDER dataset is transformed into Langfuse
    dataset items, including filtering, versioning, and metadata enrichment.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Langfuse dataset name (e.g., 'financial_qa_benchmark')",
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable description of the dataset purpose",
    )

    filter_type: DatasetFilterType = Field(
        default=DatasetFilterType.ALL,
        description="Filtering criteria for dataset subset selection",
    )

    filter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters for filtering (e.g., {'category': 'equity', 'min_complexity': 3})",
    )

    version_tag: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Version identifier for tracking dataset evolution",
    )

    include_metadata: bool = Field(
        default=True,
        description="Whether to include FinDER metadata (domain terms, ambiguity, etc.)",
    )

    metadata_enrichment: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata to attach to dataset (author, date, type)",
    )

    max_items: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of items to upload (for testing/sampling)",
    )

    @field_validator("filter_params")
    @classmethod
    def validate_filter_params(
        cls, v: Optional[Dict[str, Any]], info
    ) -> Optional[Dict[str, Any]]:
        """Validate filter_params matches filter_type requirements."""
        if v is None:
            return v

        filter_type = info.data.get("filter_type")

        if filter_type == DatasetFilterType.BY_CATEGORY:
            if "category" not in v:
                raise ValueError(
                    "filter_params must include 'category' for BY_CATEGORY filter"
                )

        elif filter_type == DatasetFilterType.BY_QUERY_TYPE:
            if "query_type" not in v:
                raise ValueError(
                    "filter_params must include 'query_type' for BY_QUERY_TYPE filter"
                )

        elif filter_type == DatasetFilterType.BY_COMPLEXITY:
            if "min_complexity" not in v and "max_complexity" not in v:
                raise ValueError(
                    "filter_params must include 'min_complexity' or 'max_complexity'"
                )

        return v

    class Config:
        use_enum_values = True


class DatasetItemMapping(BaseModel):
    """
    Mapping specification for transforming FinDER Query to Langfuse dataset item.

    Langfuse dataset items have three core fields:
    - input: Any Python object (query context)
    - expected_output: Any Python object (ground truth answer)
    - metadata: Optional dict for additional context
    """

    input_fields: List[str] = Field(
        default=["text", "id"], description="Query fields to include in input object"
    )

    expected_output_field: str = Field(
        default="expected_answer", description="Query field to use as expected_output"
    )

    metadata_fields: List[str] = Field(
        default=[
            "expected_evidence",
            "metadata.domain_term_count",
            "metadata.has_ambiguity",
            "metadata.query_type",
            "metadata.financial_subdomain",
        ],
        description="Query fields to include in metadata (supports dot notation)",
    )

    include_query_id: bool = Field(
        default=True,
        description="Whether to include query ID in metadata for traceability",
    )

    custom_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional static metadata to attach to all items"
    )

    def transform_query(self, query: Any) -> Dict[str, Any]:
        """
        Transform a FinDER Query into a Langfuse dataset item.

        Args:
            query: FinDER Query object

        Returns:
            Dict with 'input', 'expected_output', and 'metadata' keys
        """
        # Build input object
        input_obj = {}
        for field in self.input_fields:
            if hasattr(query, field):
                input_obj[field] = getattr(query, field)

        # Get expected output
        expected_output = getattr(query, self.expected_output_field, None)

        # Build metadata
        metadata = {}

        if self.include_query_id:
            metadata["query_id"] = query.id

        for field_path in self.metadata_fields:
            value = self._get_nested_field(query, field_path)
            if value is not None:
                # Flatten nested paths (e.g., "metadata.query_type" -> "query_type")
                key = field_path.split(".")[-1]
                metadata[key] = value

        if self.custom_metadata:
            metadata.update(self.custom_metadata)

        return {
            "input": input_obj,
            "expected_output": expected_output,
            "metadata": metadata,
        }

    @staticmethod
    def _get_nested_field(obj: Any, field_path: str) -> Any:
        """Get value from nested field using dot notation."""
        parts = field_path.split(".")
        value = obj
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value


class ScoreDataType(str, Enum):
    """Supported Langfuse score data types."""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"


class EvaluationScore(BaseModel):
    """
    Langfuse evaluation score with validation.

    Represents a single evaluation metric that can be attached to traces,
    observations, sessions, or dataset runs.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Score name (e.g., 'token_f1', 'hallucination', 'user_feedback')",
    )

    value: float = Field(
        ...,
        description="Numeric score value (required even for categorical/boolean types)",
    )

    data_type: ScoreDataType = Field(
        default=ScoreDataType.NUMERIC,
        description="Score data type (NUMERIC, CATEGORICAL, or BOOLEAN)",
    )

    string_value: Optional[str] = Field(
        default=None,
        max_length=200,
        description="String representation for categorical scores (e.g., 'correct', 'incorrect')",
    )

    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable explanation or reasoning for the score",
    )

    trace_id: Optional[str] = Field(
        default=None, description="Trace ID if attaching to specific trace"
    )

    observation_id: Optional[str] = Field(
        default=None,
        description="Observation ID if attaching to specific span/generation",
    )

    config_id: Optional[str] = Field(
        default=None, description="Score config ID for schema enforcement"
    )

    @field_validator("value")
    @classmethod
    def validate_value_range(cls, v: float, info) -> float:
        """Validate value is appropriate for data type."""
        data_type = info.data.get("data_type", ScoreDataType.NUMERIC)

        if data_type == ScoreDataType.BOOLEAN:
            if v not in {0.0, 1.0}:
                raise ValueError("BOOLEAN scores must have value 0.0 or 1.0")

        return v

    @field_validator("string_value")
    @classmethod
    def validate_string_value(cls, v: Optional[str], info) -> Optional[str]:
        """Validate string_value is provided for categorical scores."""
        data_type = info.data.get("data_type")

        if data_type == ScoreDataType.CATEGORICAL and v is None:
            raise ValueError("CATEGORICAL scores must include string_value")

        return v

    class Config:
        use_enum_values = True


class ItemExecutionResult(BaseModel):
    """Result of executing task on a single dataset item."""

    item_id: str = Field(..., description="Dataset item identifier")

    input: Dict[str, Any] = Field(..., description="Input passed to task function")

    output: Dict[str, Any] = Field(
        ..., description="Output returned from task function"
    )

    expected_output: Optional[str] = Field(
        default=None, description="Expected output from dataset item"
    )

    evaluations: List[EvaluationScore] = Field(
        default_factory=list, description="Item-level evaluation scores"
    )

    trace_id: str = Field(..., description="Langfuse trace ID for this item execution")

    latency_seconds: float = Field(
        ..., ge=0.0, description="Execution time for this item"
    )

    error: Optional[str] = Field(
        default=None, description="Error message if item execution failed"
    )


class ExperimentRunOutput(BaseModel):
    """
    Complete output from a Langfuse experiment run.

    Contains all item execution results, aggregate scores, and metadata
    for analyzing experiment performance.
    """

    run_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this experiment run (from Langfuse)",
    )

    experiment_name: str = Field(
        ..., min_length=1, description="Experiment name (from config)"
    )

    dataset_name: Optional[str] = Field(
        default=None, description="Langfuse dataset name if used"
    )

    dataset_run_id: Optional[str] = Field(
        default=None, description="Langfuse dataset run ID for linking results"
    )

    config: Dict[str, Any] = Field(
        ..., description="Full experiment configuration used for this run"
    )

    item_results: List[ItemExecutionResult] = Field(
        default_factory=list, description="Results for each dataset item"
    )

    aggregate_scores: Dict[str, float] = Field(
        default_factory=dict, description="Run-level aggregate evaluation scores"
    )

    total_items: int = Field(
        ..., ge=0, description="Total number of items in experiment"
    )

    successful_items: int = Field(
        ..., ge=0, description="Number of successfully processed items"
    )

    failed_items: int = Field(..., ge=0, description="Number of failed items")

    start_time: datetime = Field(..., description="Experiment start timestamp")

    end_time: Optional[datetime] = Field(
        default=None, description="Experiment end timestamp"
    )

    total_latency_seconds: float = Field(
        ..., ge=0.0, description="Total experiment execution time"
    )

    status: str = Field(
        ..., description="Experiment status (completed, failed, partial)"
    )

    langfuse_url: Optional[str] = Field(
        default=None, description="URL to view experiment in Langfuse UI"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of allowed values."""
        valid_statuses = {"completed", "failed", "partial"}
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v

    def get_success_rate(self) -> float:
        """Calculate percentage of successful items."""
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items

    def get_average_latency(self) -> float:
        """Calculate average latency per item."""
        if self.successful_items == 0:
            return 0.0
        return self.total_latency_seconds / self.successful_items

    def get_score_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all evaluation scores.

        Returns:
            Dict with aggregate scores and per-item statistics
        """
        return {
            "aggregate": self.aggregate_scores,
            "success_rate": self.get_success_rate(),
            "average_latency": self.get_average_latency(),
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
        }
