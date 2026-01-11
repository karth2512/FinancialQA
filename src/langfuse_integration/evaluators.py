"""
Evaluation functions for Langfuse experiments.

This module provides item-level and run-level evaluators for assessing
RAG pipeline performance on financial QA tasks.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from langfuse import Evaluation

from ..config.experiments import LangfuseExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Item-Level Evaluators (Per-Query Metrics)
# =============================================================================


def evaluate_token_f1(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """
    Compute token-level F1 score between generated and expected answer.

    Args:
        input: Dataset item input
        output: Task function output
        expected_output: Ground truth answer
        metadata: Dataset item metadata
        **kwargs: Additional context

    Returns:
        Evaluation with token F1 score
    """
    if not expected_output or not output:
        return Evaluation(
            name="token_f1",
            value=0.0,
            data_type="NUMERIC",
            comment="Missing expected output or generated answer",
        )

    generated_answer = output.get("answer", "")

    # Tokenize
    generated_tokens = set(generated_answer.lower().split())
    expected_tokens = set(expected_output.lower().split())

    if not expected_tokens:
        return Evaluation(name="token_f1", value=0.0, data_type="NUMERIC")

    # Calculate F1
    true_positives = len(generated_tokens & expected_tokens)
    precision = true_positives / len(generated_tokens) if generated_tokens else 0
    recall = true_positives / len(expected_tokens) if expected_tokens else 0

    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return Evaluation(
        name="token_f1",
        value=round(f1, 4),
        data_type="NUMERIC",
        comment=f"P={precision:.2f}, R={recall:.2f}",
    )


def evaluate_exact_match(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """
    Check if generated answer exactly matches expected output.

    Args:
        input: Dataset item input
        output: Task function output
        expected_output: Ground truth answer
        metadata: Dataset item metadata
        **kwargs: Additional context

    Returns:
        Evaluation with exact match score (0 or 1)
    """
    if not expected_output or not output:
        return Evaluation(
            name="exact_match", value=0.0, data_type="BOOLEAN", string_value="no"
        )

    generated_answer = output.get("answer", "").strip().lower()
    expected = expected_output.strip().lower()

    match = generated_answer == expected

    return Evaluation(
        name="exact_match",
        value=1.0 if match else 0.0,
        data_type="BOOLEAN",
        string_value="yes" if match else "no",
        comment="Exact match" if match else "No exact match",
    )


class SemanticSimilarityEvaluator:
    """
    Evaluator for semantic similarity using sentence embeddings.

    Uses simple embedding cosine similarity (fast but less accurate).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with embedding model.

        Args:
            model_name: Sentence transformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            self.model = None

    def __call__(
        self,
        *,
        input: Dict[str, Any],
        output: Dict[str, Any],
        expected_output: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Evaluation:
        """
        Compute semantic similarity between answers.

        Args:
            input: Dataset item input
            output: Task function output
            expected_output: Ground truth answer
            metadata: Dataset item metadata
            **kwargs: Additional context

        Returns:
            Evaluation with semantic similarity score
        """
        if self.model is None:
            return Evaluation(
                name="semantic_similarity",
                value=0.0,
                data_type="NUMERIC",
                comment="sentence-transformers not available",
            )

        if not expected_output or not output:
            return Evaluation(
                name="semantic_similarity", value=0.0, data_type="NUMERIC"
            )

        generated_answer = output.get("answer", "")

        # Compute embeddings
        embeddings = self.model.encode([generated_answer, expected_output])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return Evaluation(
            name="semantic_similarity",
            value=round(float(similarity), 4),
            data_type="NUMERIC",
            comment=f"Cosine similarity: {similarity:.4f}",
        )


def evaluate_retrieval_precision(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """
    Compute retrieval precision using expected evidence from metadata.

    Args:
        input: Dataset item input
        output: Task function output (with retrieved_passages)
        expected_output: Ground truth answer
        metadata: Dataset item metadata (with expected_evidence)
        **kwargs: Additional context

    Returns:
        Evaluation with retrieval precision score
    """
    if not metadata or "expected_evidence" not in metadata:
        return Evaluation(
            name="retrieval_precision",
            value=0.0,
            data_type="NUMERIC",
            comment="No expected evidence in metadata",
        )

    expected_evidence = set(metadata["expected_evidence"])
    retrieved_passages = output.get("retrieved_passages", [])

    if not retrieved_passages:
        return Evaluation(name="retrieval_precision", value=0.0, data_type="NUMERIC")

    # Count relevant retrieved passages
    relevant_count = sum(
        1 for passage in retrieved_passages if passage.get("text") in expected_evidence
    )

    precision = relevant_count / len(retrieved_passages)

    return Evaluation(
        name="retrieval_precision",
        value=round(precision, 4),
        data_type="NUMERIC",
        comment=f"{relevant_count}/{len(retrieved_passages)} relevant",
    )


def evaluate_retrieval_recall(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """
    Compute retrieval recall using expected evidence from metadata.

    Args:
        input: Dataset item input
        output: Task function output (with retrieved_passages)
        expected_output: Ground truth answer
        metadata: Dataset item metadata (with expected_evidence)
        **kwargs: Additional context

    Returns:
        Evaluation with retrieval recall score
    """
    if not metadata or "expected_evidence" not in metadata:
        return Evaluation(
            name="retrieval_recall",
            value=0.0,
            data_type="NUMERIC",
            comment="No expected evidence in metadata",
        )

    expected_evidence = set(metadata["expected_evidence"])
    retrieved_passages = output.get("retrieved_passages", [])

    if not expected_evidence:
        return Evaluation(name="retrieval_recall", value=0.0, data_type="NUMERIC")

    # Count expected evidence found in retrieved passages
    retrieved_texts = {p.get("text") for p in retrieved_passages}
    found_count = len(expected_evidence & retrieved_texts)

    recall = found_count / len(expected_evidence)

    return Evaluation(
        name="retrieval_recall",
        value=round(recall, 4),
        data_type="NUMERIC",
        comment=f"{found_count}/{len(expected_evidence)} expected found",
    )


def evaluate_retrieval_quality(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """
    Combined retrieval quality evaluator (F1 of precision and recall).

    Args:
        input: Dataset item input
        output: Task function output
        expected_output: Ground truth answer
        metadata: Dataset item metadata
        **kwargs: Additional context

    Returns:
        Evaluation with retrieval quality F1 score
    """
    # Get precision and recall
    prec_eval = evaluate_retrieval_precision(
        input=input,
        output=output,
        expected_output=expected_output,
        metadata=metadata,
        **kwargs,
    )
    rec_eval = evaluate_retrieval_recall(
        input=input,
        output=output,
        expected_output=expected_output,
        metadata=metadata,
        **kwargs,
    )

    precision = prec_eval.value
    recall = rec_eval.value

    # Calculate F1
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return Evaluation(
        name="retrieval_quality",
        value=round(f1, 4),
        data_type="NUMERIC",
        comment=f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}",
    )


# =============================================================================
# Run-Level Evaluators (Aggregate Metrics)
# =============================================================================


def compute_average_accuracy(*, item_results: List[Any], **kwargs) -> Evaluation:
    """
    Compute average token F1 score across all queries.

    Args:
        item_results: List of item execution results
        **kwargs: Additional context

    Returns:
        Evaluation with average accuracy score
    """
    f1_scores = []

    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name == "token_f1":
                f1_scores.append(evaluation.value)

    if not f1_scores:
        return Evaluation(
            name="average_accuracy",
            value=0.0,
            data_type="NUMERIC",
            comment="No token_f1 scores found",
        )

    avg_f1 = sum(f1_scores) / len(f1_scores)

    return Evaluation(
        name="average_accuracy",
        value=round(avg_f1, 4),
        data_type="NUMERIC",
        comment=f"Mean F1 over {len(f1_scores)} queries",
    )


def compute_aggregate_retrieval_metrics(
    *, item_results: List[Any], **kwargs
) -> Evaluation:
    """
    Compute aggregate retrieval metrics (precision, recall, F1).

    Args:
        item_results: List of item execution results
        **kwargs: Additional context

    Returns:
        Evaluation with aggregate retrieval F1 score
    """
    precisions = []
    recalls = []

    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name == "retrieval_precision":
                precisions.append(evaluation.value)
            elif evaluation.name == "retrieval_recall":
                recalls.append(evaluation.value)

    if not precisions or not recalls:
        return Evaluation(
            name="aggregate_retrieval_metrics",
            value=0.0,
            data_type="NUMERIC",
            comment="Missing retrieval metrics",
        )

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_f1 = (
        2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        if (mean_precision + mean_recall) > 0
        else 0
    )

    return Evaluation(
        name="aggregate_retrieval_metrics",
        value=round(mean_f1, 4),
        data_type="NUMERIC",
        comment=f"P={mean_precision:.4f}, R={mean_recall:.4f}, F1={mean_f1:.4f}",
    )


class PassRateEvaluator:
    """
    Run-level evaluator for pass rate based on thresholds.
    """

    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize with quality thresholds.

        Args:
            thresholds: Dict mapping metric names to minimum passing values
        """
        self.thresholds = thresholds

    def __call__(self, *, item_results: List[Any], **kwargs) -> Evaluation:
        """
        Compute percentage of queries meeting thresholds.

        Args:
            item_results: List of item execution results
            **kwargs: Additional context

        Returns:
            Evaluation with pass rate score
        """
        passed_count = 0
        total_count = len(item_results)

        for result in item_results:
            item_passed = True

            for metric_name, threshold in self.thresholds.items():
                metric_found = False

                for evaluation in result.evaluations:
                    if evaluation.name == metric_name:
                        metric_found = True
                        if evaluation.value < threshold:
                            item_passed = False
                            break

                if not metric_found:
                    item_passed = False
                    break

            if item_passed:
                passed_count += 1

        pass_rate = passed_count / total_count if total_count > 0 else 0

        return Evaluation(
            name="pass_rate",
            value=round(pass_rate, 4),
            data_type="NUMERIC",
            comment=f"{passed_count}/{total_count} queries passed thresholds",
        )


# =============================================================================
# Evaluator Registration
# =============================================================================


def register_evaluators(config: LangfuseExperimentConfig) -> Tuple[List, List]:
    """
    Register item-level and run-level evaluators based on config.

    Args:
        config: Experiment configuration with evaluator names

    Returns:
        Tuple of (item_evaluators, run_evaluators)
    """
    logger.info("Registering evaluators from config")

    # Map evaluator names to implementations
    item_evaluator_map = {
        "token_f1": evaluate_token_f1,
        "exact_match": evaluate_exact_match,
        "semantic_similarity": SemanticSimilarityEvaluator(),
        "retrieval_precision": evaluate_retrieval_precision,
        "retrieval_recall": evaluate_retrieval_recall,
        "retrieval_quality": evaluate_retrieval_quality,
    }

    run_evaluator_map = {
        "average_accuracy": compute_average_accuracy,
        "aggregate_retrieval_metrics": compute_aggregate_retrieval_metrics,
        "pass_rate": PassRateEvaluator(config.evaluation_thresholds),
    }

    # Build evaluator lists from config
    item_evaluators = []
    for name in config.item_evaluator_names:
        if name in item_evaluator_map:
            item_evaluators.append(item_evaluator_map[name])
            logger.info(f"Registered item evaluator: {name}")
        else:
            logger.warning(f"Unknown item evaluator: {name}")

    run_evaluators = []
    for name in config.run_evaluator_names:
        if name in run_evaluator_map:
            run_evaluators.append(run_evaluator_map[name])
            logger.info(f"Registered run evaluator: {name}")
        else:
            logger.warning(f"Unknown run evaluator: {name}")

    logger.info(
        f"Registered {len(item_evaluators)} item evaluators, {len(run_evaluators)} run evaluators"
    )

    return item_evaluators, run_evaluators
