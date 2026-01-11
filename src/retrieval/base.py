"""
Base retrieval interface for all retrieval strategies.

This module defines the abstract RetrieverBase class that all concrete
retrieval implementations must extend.
"""

from abc import ABC, abstractmethod
from typing import List
from src.data_handler.models import EvidencePassage, RetrievalResult


class RetrieverBase(ABC):
    """Abstract base class for all retrieval strategies."""

    def __init__(self, config: dict):
        """
        Initialize retriever with configuration.

        Args:
            config: Retrieval configuration dict
        """
        self.config = config
        self.is_indexed = False

    @abstractmethod
    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """
        Build index over evidence corpus (one-time setup).

        Args:
            passages: All evidence passages to be indexed

        Raises:
            RetrievalError: If indexing fails
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve top-K relevant passages for a query.

        Args:
            query_text: Query text (possibly disambiguated)
            top_k: Number of passages to retrieve

        Returns:
            RetrievalResult with ranked passages

        Raises:
            IndexNotBuiltError: If index not built
            RetrievalError: If retrieval fails
        """
        pass


class RetrievalError(Exception):
    """Raised when retrieval execution fails."""

    def __init__(self, strategy: str, error_message: str, query_text: str):
        """
        Initialize retrieval error.

        Args:
            strategy: Which retrieval strategy failed
            error_message: Error description
            query_text: Query that caused the error
        """
        self.strategy = strategy
        self.error_message = error_message
        self.query_text = query_text
        super().__init__(
            f"Retrieval failed ({strategy}): {error_message} for query: {query_text}"
        )


class IndexNotBuiltError(Exception):
    """Raised when attempting retrieval before building index."""

    def __init__(self, strategy: str):
        """
        Initialize index not built error.

        Args:
            strategy: Which retrieval strategy has no index
        """
        self.strategy = strategy
        super().__init__(f"Index not built for {strategy}. Call index_corpus() first.")
