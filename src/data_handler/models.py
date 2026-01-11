"""
Data models for the FinDER Multi-Agent RAG System.

This module defines the core data structures used throughout the system,
including queries, evidence passages, and retrieval results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QueryMetadata:
    """Metadata about query complexity and characteristics."""

    domain_term_count: int
    has_ambiguity: bool
    query_type: str  # "factual", "temporal_comparison", "trend_analysis", "definition", or FinDER types
    required_evidence_count: int
    financial_subdomain: (
        str  # "equity", "fixed_income", "derivatives", "corporate_finance"
    )
    reasoning_required: bool = False  # Whether multi-hop reasoning is needed

    def __post_init__(self):
        """Validate metadata fields."""
        if self.domain_term_count < 0:
            raise ValueError("domain_term_count must be non-negative")
        if self.required_evidence_count < 1:
            raise ValueError("required_evidence_count must be positive")


@dataclass
class Query:
    """A financial question from the FinDER dataset."""

    id: str  # Unique identifier (dataset index or UUID)
    text: str  # Question text
    expected_answer: str  # Ground truth answer from dataset
    expected_evidence: List[str]  # Annotated relevant passages
    metadata: QueryMetadata  # Additional query characteristics

    def __post_init__(self):
        """Validate query fields."""
        if not self.id:
            raise ValueError("id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not self.expected_answer.strip():
            raise ValueError("expected_answer must be non-empty")
        if not self.expected_evidence:
            raise ValueError("expected_evidence must be non-empty list")


@dataclass
class EvidenceMetadata:
    """Metadata about evidence passage provenance and authority."""

    source_type: str  # "10-K", "earnings_report", "regulatory_filing", "news"
    publication_date: Optional[str] = None  # ISO 8601 date string
    page_number: Optional[int] = None
    section: Optional[str] = None

    def __post_init__(self):
        """Validate metadata fields."""
        valid_types = {
            "10-K",
            "earnings_report",
            "regulatory_filing",
            "news",
            "unknown",
        }
        if self.source_type not in valid_types:
            raise ValueError(f"source_type must be one of {valid_types}")


@dataclass
class EvidencePassage:
    """A text passage from the financial corpus."""

    id: str  # Unique passage identifier
    text: str  # Passage content
    document_id: str  # Source document identifier
    metadata: EvidenceMetadata  # Passage provenance

    def __post_init__(self):
        """Validate evidence passage fields."""
        if not self.id:
            raise ValueError("id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not self.document_id:
            raise ValueError("document_id must reference a valid document")


@dataclass
class RetrievedPassage:
    """A single retrieved passage with relevance score."""

    passage: EvidencePassage  # The passage itself
    score: float  # Relevance score (higher = more relevant)
    rank: int  # Position in retrieval results (1-indexed)
    is_relevant: Optional[bool] = None  # Whether in expected_evidence (for evaluation)

    def __post_init__(self):
        """Validate retrieved passage fields."""
        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if self.rank < 1:
            raise ValueError("rank must be positive (1-indexed)")


@dataclass
class RetrievalResult:
    """Result of retrieving evidence passages for a query."""

    query_id: str  # Reference to Query
    retrieved_passages: List[RetrievedPassage]  # Ranked list of passages
    strategy: str  # "bm25", "dense", "hybrid"
    retrieval_time_seconds: float  # Latency
    metadata: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific metadata

    def __post_init__(self):
        """Validate retrieval result fields."""
        if not self.query_id:
            raise ValueError("query_id must reference an existing Query")
        if not self.retrieved_passages:
            raise ValueError("retrieved_passages must be non-empty list")
        if self.strategy not in {"bm25", "dense", "hybrid"}:
            raise ValueError("strategy must be one of: bm25, dense, hybrid")
        if self.retrieval_time_seconds < 0:
            raise ValueError("retrieval_time_seconds must be non-negative")
