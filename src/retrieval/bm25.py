"""
BM25 keyword-based retrieval implementation.

This module implements BM25 retrieval using the rank_bm25 library.
"""

from typing import List
import time
from rank_bm25 import BM25Okapi
from src.retrieval.base import RetrieverBase, IndexNotBuiltError, RetrievalError
from src.data_handler.models import EvidencePassage, RetrievalResult, RetrievedPassage


class BM25Retriever(RetrieverBase):
    """BM25 keyword-based retrieval using rank_bm25 library."""

    def __init__(self, config: dict):
        """
        Initialize BM25 retriever.

        Args:
            config: Configuration dict with optional:
                - k1: BM25 term frequency saturation (default: 1.5)
                - b: BM25 length normalization (default: 0.75)
        """
        super().__init__(config)
        self.k1 = config.get("k1", 1.5)
        self.b = config.get("b", 0.75)
        self.bm25 = None
        self.passages = []
        self.tokenized_corpus = []

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """Build BM25 index from passages."""
        try:
            self.passages = passages
            # Tokenize corpus (simple whitespace tokenization)
            self.tokenized_corpus = [
                passage.text.lower().split() for passage in passages
            ]

            # Build BM25 index
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

            self.is_indexed = True
            print(f"✓ BM25 index built with {len(passages)} passages")

        except Exception as e:
            raise RetrievalError("bm25", f"Failed to build index: {e}", "")

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-K passages using BM25 scoring."""
        if not self.is_indexed or self.bm25 is None:
            raise IndexNotBuiltError("bm25")

        start_time = time.time()

        try:
            # Tokenize query
            tokenized_query = query_text.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-K indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            # Create retrieved passages
            retrieved_passages = []
            for rank, idx in enumerate(top_indices, start=1):
                retrieved_passage = RetrievedPassage(
                    passage=self.passages[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    is_relevant=None,  # Will be set during evaluation
                )
                retrieved_passages.append(retrieved_passage)

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query_id=query_text,  # Will be set by caller
                retrieved_passages=retrieved_passages,
                strategy="bm25",
                retrieval_time_seconds=retrieval_time,
                metadata={"k1": self.k1, "b": self.b},
            )

        except Exception as e:
            raise RetrievalError("bm25", f"Retrieval failed: {e}", query_text)
