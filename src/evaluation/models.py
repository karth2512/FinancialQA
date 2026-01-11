"""
Evaluation data models for tracking runs, metrics, and results.

This module defines data structures for evaluation runs, query traces,
and aggregated metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.data_handler.models import Query, RetrievalResult


@dataclass
class AnswerMetrics:
    """Answer correctness metrics."""

    exact_match_rate: float  # % of exact matches
    mean_token_f1: float  # Average token-level F1
    mean_semantic_similarity: float  # Average embedding cosine similarity
    mean_llm_judge_score: Optional[float] = None  # Average LLM-as-judge score

    def __post_init__(self):
        """Validate answer metrics."""
        for field_name in [
            "exact_match_rate",
            "mean_token_f1",
            "mean_semantic_similarity",
        ]:
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1] range")


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    mean_precision: float  # Average precision across queries
    mean_recall: float  # Average recall across queries
    mean_f1: float  # Average F1 score
    mean_mrr: float  # Mean reciprocal rank

    def __post_init__(self):
        """Validate retrieval metrics."""
        for field_name in ["mean_precision", "mean_recall", "mean_f1", "mean_mrr"]:
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1] range")


@dataclass
class OperationalMetrics:
    """Performance and cost metrics."""

    mean_latency_seconds: float  # Average end-to-end latency
    p95_latency_seconds: float  # 95th percentile latency
    total_tokens: int  # Total prompt + completion tokens
    mean_cost_per_query: float  # Average cost per query
    total_cost: float  # Total run cost

    def __post_init__(self):
        """Validate operational metrics."""
        if self.mean_latency_seconds < 0:
            raise ValueError("mean_latency_seconds must be non-negative")
        if self.p95_latency_seconds < 0:
            raise ValueError("p95_latency_seconds must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.mean_cost_per_query < 0:
            raise ValueError("mean_cost_per_query must be non-negative")
        if self.total_cost < 0:
            raise ValueError("total_cost must be non-negative")


@dataclass
class AggregateMetrics:
    """Summary metrics across all queries in evaluation run."""

    total_queries: int
    answer_metrics: AnswerMetrics
    retrieval_metrics: RetrievalMetrics
    operational_metrics: OperationalMetrics
    breakdown_by_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    breakdown_by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate aggregate metrics."""
        if self.total_queries < 0:
            raise ValueError("total_queries must be non-negative")


@dataclass
class QueryTrace:
    """Detailed trace of processing a single query."""

    query: Query  # The input query
    retrieval_result: RetrievalResult  # Retrieved evidence
    generated_answer: str  # Final generated answer
    correctness_metrics: Dict[str, float]  # Answer correctness scores
    retrieval_quality_metrics: Dict[str, float]  # Retrieval quality scores
    latency_seconds: float  # Total latency for this query
    cost_estimate: float  # Total cost for this query
    error: Optional[str] = None  # Error message if query failed

    def __post_init__(self):
        """Validate query trace."""
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        if self.cost_estimate < 0:
            raise ValueError("cost_estimate must be non-negative")
        if not self.error and not self.generated_answer.strip():
            raise ValueError("generated_answer must be non-empty if no error occurred")


@dataclass
class EvaluationRun:
    """A complete evaluation run on the FinDER dataset."""

    run_id: str  # Unique identifier (e.g., "v0.0.0-baseline", "exp-2026-01-09-001")
    version: Optional[str]  # Semantic version if applicable
    timestamp: str  # ISO 8601 start time
    config: Dict[str, Any]  # Full configuration for this run
    dataset_version: str  # FinDER dataset version/commit hash
    query_traces: List[QueryTrace]  # Individual query results
    aggregate_metrics: Optional[AggregateMetrics]  # Summary metrics
    status: str  # "running", "completed", "failed"

    def __post_init__(self):
        """Validate evaluation run."""
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.dataset_version:
            raise ValueError("dataset_version must be non-empty")
        if self.status not in {"running", "completed", "failed"}:
            raise ValueError("status must be one of: running, completed, failed")
