"""
Hybrid retrieval combining BM25 and dense retrieval with rank fusion.

This module implements hybrid search that combines keyword and semantic retrieval.
"""

from typing import List
import time
from src.retrieval.base import RetrieverBase, IndexNotBuiltError, RetrievalError
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.data.models import EvidencePassage, RetrievalResult, RetrievedPassage


class HybridRetriever(RetrieverBase):
    """Hybrid retrieval combining BM25 and dense strategies."""

    def __init__(self, config: dict):
        """
        Initialize hybrid retriever.

        Args:
            config: Configuration dict with:
                - bm25_weight: Weight for BM25 scores (0-1)
                - dense_weight: Weight for dense scores (0-1)
                - embedding_model: Model for dense retrieval
                - fusion_method: "reciprocal_rank" or "score_fusion"
        """
        super().__init__(config)
        self.bm25_weight = config.get("bm25_weight", 0.5)
        self.dense_weight = config.get("dense_weight", 0.5)
        self.fusion_method = config.get("fusion_method", "reciprocal_rank")

        # Initialize both retrievers
        bm25_config = {
            "k1": config.get("k1", 1.5),
            "b": config.get("b", 0.75),
        }
        dense_config = {
            "embedding_model": config.get("embedding_model", "all-MiniLM-L6-v2"),
            "collection_name": config.get("collection_name", "finder_corpus_hybrid"),
            "similarity_metric": config.get("similarity_metric", "cosine"),
        }

        self.bm25_retriever = BM25Retriever(bm25_config)
        self.dense_retriever = DenseRetriever(dense_config)

    def index_corpus(self, passages: List[EvidencePassage]) -> None:
        """Build indexes for both BM25 and dense retrieval."""
        try:
            print("Building hybrid index (BM25 + Dense)...")

            # Index with BM25
            self.bm25_retriever.index_corpus(passages)

            # Index with dense retriever
            self.dense_retriever.index_corpus(passages)

            self.is_indexed = True
            print(f"✓ Hybrid index built with {len(passages)} passages")

        except Exception as e:
            raise RetrievalError("hybrid", f"Failed to build index: {e}", "")

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-K passages using hybrid approach."""
        if not self.is_indexed:
            raise IndexNotBuiltError("hybrid")

        start_time = time.time()

        try:
            # Retrieve from both strategies
            # Retrieve more than top_k to allow for fusion
            retrieval_k = min(top_k * 2, 20)

            bm25_results = self.bm25_retriever.retrieve(query_text, top_k=retrieval_k)
            dense_results = self.dense_retriever.retrieve(query_text, top_k=retrieval_k)

            # Fuse results
            if self.fusion_method == "reciprocal_rank":
                fused_passages = self._reciprocal_rank_fusion(
                    bm25_results, dense_results, top_k
                )
            else:  # score_fusion
                fused_passages = self._score_fusion(
                    bm25_results, dense_results, top_k
                )

            retrieval_time = time.time() - start_time

            return RetrievalResult(
                query_id="",
                retrieved_passages=fused_passages,
                strategy="hybrid",
                retrieval_time_seconds=retrieval_time,
                metadata={
                    "bm25_weight": self.bm25_weight,
                    "dense_weight": self.dense_weight,
                    "fusion_method": self.fusion_method,
                },
            )

        except Exception as e:
            raise RetrievalError("hybrid", f"Retrieval failed: {e}", query_text)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: RetrievalResult,
        dense_results: RetrievalResult,
        top_k: int,
    ) -> List[RetrievedPassage]:
        """
        Fuse results using reciprocal rank fusion.

        Score = bm25_weight * (1 / bm25_rank) + dense_weight * (1 / dense_rank)
        """
        # Create mapping of passage_id to scores
        fusion_scores = {}

        # Add BM25 scores
        for passage in bm25_results.retrieved_passages:
            passage_id = passage.passage.id
            rr_score = self.bm25_weight / passage.rank
            fusion_scores[passage_id] = {
                "score": rr_score,
                "passage": passage.passage,
                "bm25_rank": passage.rank,
                "bm25_score": passage.score,
                "dense_rank": None,
                "dense_score": None,
            }

        # Add dense scores
        for passage in dense_results.retrieved_passages:
            passage_id = passage.passage.id
            rr_score = self.dense_weight / passage.rank

            if passage_id in fusion_scores:
                fusion_scores[passage_id]["score"] += rr_score
                fusion_scores[passage_id]["dense_rank"] = passage.rank
                fusion_scores[passage_id]["dense_score"] = passage.score
            else:
                fusion_scores[passage_id] = {
                    "score": rr_score,
                    "passage": passage.passage,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": passage.rank,
                    "dense_score": passage.score,
                }

        # Sort by fused score and create ranked list
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]

        # Create RetrievedPassage objects
        fused_passages = []
        for rank, (passage_id, data) in enumerate(sorted_results, start=1):
            retrieved_passage = RetrievedPassage(
                passage=data["passage"],
                score=data["score"],
                rank=rank,
                is_relevant=None,
            )
            fused_passages.append(retrieved_passage)

        return fused_passages

    def _score_fusion(
        self,
        bm25_results: RetrievalResult,
        dense_results: RetrievalResult,
        top_k: int,
    ) -> List[RetrievedPassage]:
        """
        Fuse results using normalized score fusion.

        Normalize scores to [0, 1] then combine with weights.
        """
        # Normalize BM25 scores
        bm25_scores = [p.score for p in bm25_results.retrieved_passages]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0

        # Normalize dense scores
        dense_scores = [p.score for p in dense_results.retrieved_passages]
        max_dense = max(dense_scores) if dense_scores else 1.0
        min_dense = min(dense_scores) if dense_scores else 0.0
        dense_range = max_dense - min_dense if max_dense != min_dense else 1.0

        fusion_scores = {}

        # Add normalized BM25 scores
        for passage in bm25_results.retrieved_passages:
            passage_id = passage.passage.id
            norm_score = (passage.score - min_bm25) / bm25_range
            weighted_score = self.bm25_weight * norm_score

            fusion_scores[passage_id] = {
                "score": weighted_score,
                "passage": passage.passage,
            }

        # Add normalized dense scores
        for passage in dense_results.retrieved_passages:
            passage_id = passage.passage.id
            norm_score = (passage.score - min_dense) / dense_range
            weighted_score = self.dense_weight * norm_score

            if passage_id in fusion_scores:
                fusion_scores[passage_id]["score"] += weighted_score
            else:
                fusion_scores[passage_id] = {
                    "score": weighted_score,
                    "passage": passage.passage,
                }

        # Sort and create ranked list
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]

        fused_passages = []
        for rank, (passage_id, data) in enumerate(sorted_results, start=1):
            retrieved_passage = RetrievedPassage(
                passage=data["passage"],
                score=data["score"],
                rank=rank,
                is_relevant=None,
            )
            fused_passages.append(retrieved_passage)

        return fused_passages
