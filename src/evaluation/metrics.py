"""
Evaluation metrics calculator for answer correctness and retrieval quality.

This module implements comprehensive metrics including F1, semantic similarity,
precision, recall, and operational metrics.
"""

from typing import List, Dict
import numpy as np


class EvaluationMetrics:
    """Calculator for answer and retrieval quality metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        try:
            from sentence_transformers import SentenceTransformer

            self.sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print(
                "Warning: sentence-transformers not available, semantic similarity disabled"
            )
            self.sem_model = None

    def compute_answer_metrics(
        self,
        generated: str,
        expected: str,
        include_llm_judge: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate answer correctness metrics.

        Args:
            generated: Generated answer text
            expected: Expected answer text
            include_llm_judge: Whether to compute LLM-as-judge score

        Returns:
            Dict with keys: exact_match, token_f1, semantic_similarity
        """
        # Exact match
        exact_match = float(generated.strip().lower() == expected.strip().lower())

        # Token-level F1
        token_f1 = self._compute_token_f1(generated, expected)

        # Semantic similarity
        if self.sem_model is not None:
            semantic_sim = self._compute_semantic_similarity(generated, expected)
        else:
            semantic_sim = 0.0

        metrics = {
            "exact_match": exact_match,
            "token_f1": token_f1,
            "semantic_similarity": semantic_sim,
        }

        if include_llm_judge:
            # TODO: Implement LLM-as-judge (optional, requires LLM call)
            metrics["llm_judge_score"] = None

        return metrics

    def compute_retrieval_metrics(
        self,
        retrieved_passage_ids: List[str],
        relevant_passage_ids: List[str],
    ) -> Dict[str, float]:
        """
        Calculate retrieval quality metrics.

        Args:
            retrieved_passage_ids: IDs of retrieved passages
            relevant_passage_ids: IDs of ground truth relevant passages

        Returns:
            Dict with keys: precision, recall, f1, mrr
        """
        retrieved_set = set(retrieved_passage_ids)
        relevant_set = set(relevant_passage_ids)

        # True positives
        tp = len(retrieved_set & relevant_set)

        # Precision and Recall
        precision = tp / len(retrieved_set) if retrieved_set else 0.0
        recall = tp / len(relevant_set) if relevant_set else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Mean Reciprocal Rank
        mrr = 0.0
        for rank, passage_id in enumerate(retrieved_passage_ids, start=1):
            if passage_id in relevant_set:
                mrr = 1.0 / rank
                break

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
        }

    def compute_aggregate_metrics(
        self,
        answer_metrics_list: List[Dict[str, float]],
        retrieval_metrics_list: List[Dict[str, float]],
        latencies: List[float],
        costs: List[float],
    ) -> Dict[str, any]:
        """
        Aggregate metrics across all queries.

        Args:
            answer_metrics_list: List of answer metrics dicts
            retrieval_metrics_list: List of retrieval metrics dicts
            latencies: List of latency values
            costs: List of cost estimates

        Returns:
            Dict with aggregate statistics
        """
        # Answer metrics
        answer_agg = {
            "exact_match_rate": np.mean(
                [m["exact_match"] for m in answer_metrics_list]
            ),
            "mean_token_f1": np.mean([m["token_f1"] for m in answer_metrics_list]),
            "mean_semantic_similarity": np.mean(
                [m.get("semantic_similarity", 0) for m in answer_metrics_list]
            ),
        }

        # Retrieval metrics
        retrieval_agg = {
            "mean_precision": np.mean([m["precision"] for m in retrieval_metrics_list]),
            "mean_recall": np.mean([m["recall"] for m in retrieval_metrics_list]),
            "mean_f1": np.mean([m["f1"] for m in retrieval_metrics_list]),
            "mean_mrr": np.mean([m["mrr"] for m in retrieval_metrics_list]),
        }

        # Operational metrics
        operational_agg = {
            "mean_latency_seconds": np.mean(latencies),
            "p95_latency_seconds": np.percentile(latencies, 95),
            "mean_cost_per_query": np.mean(costs),
            "total_cost": np.sum(costs),
        }

        return {
            "answer": answer_agg,
            "retrieval": retrieval_agg,
            "operational": operational_agg,
        }

    def _compute_token_f1(self, generated: str, expected: str) -> float:
        """Compute token-level F1 score."""
        gen_tokens = set(generated.lower().split())
        exp_tokens = set(expected.lower().split())

        if not gen_tokens or not exp_tokens:
            return 0.0

        common = gen_tokens & exp_tokens

        if not common:
            return 0.0

        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(exp_tokens)

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if self.sem_model is None:
            return 0.0

        try:
            embeddings = self.sem_model.encode([text1, text2])
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return 0.0
