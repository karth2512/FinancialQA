"""
Evaluation runner for orchestrating evaluation runs on the FinDER dataset.

This module provides the main entry point for running evaluations with
baseline or multi-agent RAG pipelines.
"""

import time
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.config.experiments import ExperimentConfig
from src.data.loader import FinDERLoader
from src.data.indexer import IndexBuilder
from src.data.models import Query
from src.rag.baseline import BaselineRAG
from src.evaluation.models import (
    EvaluationRun,
    QueryTrace,
    AggregateMetrics,
    AnswerMetrics,
    RetrievalMetrics,
    OperationalMetrics,
)
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.logger import LangfuseLogger


class EvaluationRunner:
    """Orchestrates evaluation runs on the FinDER dataset."""

    def __init__(
        self,
        config: ExperimentConfig,
        langfuse_enabled: bool = True,
    ):
        """
        Initialize evaluation runner.

        Args:
            config: Experiment configuration
            langfuse_enabled: Whether to log to Langfuse
        """
        self.config = config
        self.metrics_calculator = EvaluationMetrics()
        self.langfuse_logger = LangfuseLogger(enabled=langfuse_enabled)

    def run_evaluation(
        self,
        dataset_subset: Optional[int] = None,
    ) -> EvaluationRun:
        """
        Execute complete evaluation run on dataset.

        Args:
            dataset_subset: If provided, evaluate only first N queries

        Returns:
            Complete EvaluationRun with all traces and metrics
        """
        print(f"\n{'='*70}")
        print(f"Starting Evaluation Run: {self.config.run_id}")
        print(f"{'='*70}\n")

        # Load dataset
        print("Loading FinDER dataset...")
        loader = FinDERLoader()
        queries = loader.load()

        if dataset_subset:
            queries = queries[:dataset_subset]
            print(f"Using subset: {len(queries)} queries")

        # Load corpus and build index if needed
        print("Loading corpus...")
        corpus = loader.load_corpus()

        # Build indexes based on retrieval strategy
        self._ensure_index_built(corpus)

        # Get dataset version
        dataset_version = loader.get_dataset_version()

        # Create RAG pipeline based on configuration
        print(f"Initializing {self.config.pipeline_type} pipeline...")
        pipeline = self._create_pipeline()

        # Create evaluation run
        evaluation_run = EvaluationRun(
            run_id=self.config.run_id,
            version=None,  # Could parse from run_id if semantic version
            timestamp=datetime.utcnow().isoformat(),
            config=self.config.model_dump(),
            dataset_version=dataset_version,
            query_traces=[],
            aggregate_metrics=None,
            status="running",
        )

        # Process all queries
        print(f"\nProcessing {len(queries)} queries...")
        query_traces = []

        for query in tqdm(queries, desc="Evaluating queries"):
            try:
                trace = self._process_single_query(query, pipeline)
                query_traces.append(trace)
            except Exception as e:
                print(f"\nError processing query {query.id}: {e}")
                # Create error trace
                error_trace = self._create_error_trace(query, str(e))
                query_traces.append(error_trace)

        evaluation_run.query_traces = query_traces

        # Compute aggregate metrics
        print("\nComputing aggregate metrics...")
        aggregate_metrics = self._compute_aggregate_metrics(query_traces)
        evaluation_run.aggregate_metrics = aggregate_metrics
        evaluation_run.status = "completed"

        # Log to Langfuse
        if self.langfuse_logger.enabled:
            print("Logging to Langfuse...")
            self.langfuse_logger.log_evaluation_run(evaluation_run)

        # Print summary
        self._print_summary(evaluation_run)

        return evaluation_run

    def run_single_query(
        self,
        query: Query,
    ) -> QueryTrace:
        """
        Evaluate a single query (for debugging).

        Args:
            query: Query to evaluate

        Returns:
            QueryTrace for the query
        """
        # Create pipeline
        pipeline = self._create_pipeline()

        # Process query
        return self._process_single_query(query, pipeline)

    def _create_pipeline(self):
        """Create RAG pipeline based on configuration."""
        if self.config.pipeline_type == "baseline":
            return BaselineRAG.from_config(self.config.model_dump())
        elif self.config.pipeline_type == "multiagent":
            # TODO: Implement multi-agent pipeline
            raise NotImplementedError("Multi-agent pipeline not yet implemented")
        elif self.config.pipeline_type == "specialized":
            # TODO: Implement specialized pipeline
            raise NotImplementedError("Specialized pipeline not yet implemented")
        else:
            raise ValueError(f"Unknown pipeline type: {self.config.pipeline_type}")

    def _ensure_index_built(self, corpus):
        """Ensure retrieval index is built."""
        strategy = self.config.retrieval_config.strategy

        if strategy == "bm25":
            # BM25 builds index on-the-fly
            pass
        elif strategy == "dense":
            # Dense may use existing ChromaDB collection
            print(f"Using dense retrieval (collection: {self.config.retrieval_config.embedding_model})")
        elif strategy == "hybrid":
            print("Using hybrid retrieval (BM25 + Dense)")

    def _process_single_query(
        self,
        query: Query,
        pipeline,
    ) -> QueryTrace:
        """Process a single query and return trace."""
        # Run query through pipeline
        result = pipeline.process_query(query, top_k=self.config.retrieval_config.top_k)

        # Compute metrics
        correctness_metrics = self.metrics_calculator.compute_answer_metrics(
            generated=result["generated_answer"],
            expected=query.expected_answer,
        )

        # Get retrieved passage IDs
        retrieved_ids = [p.passage.id for p in result["retrieval_result"].retrieved_passages]
        retrieval_metrics = self.metrics_calculator.compute_retrieval_metrics(
            retrieved_passage_ids=retrieved_ids,
            relevant_passage_ids=query.expected_evidence,
        )

        # Create query trace
        trace = QueryTrace(
            query=query,
            retrieval_result=result["retrieval_result"],
            agent_executions=[],  # No agents in baseline
            generated_answer=result["generated_answer"],
            correctness_metrics=correctness_metrics,
            retrieval_quality_metrics=retrieval_metrics,
            latency_seconds=result["latency_seconds"],
            cost_estimate=result["cost_estimate"],
            error=None,
        )

        return trace

    def _create_error_trace(self, query: Query, error_message: str) -> QueryTrace:
        """Create error trace for failed query."""
        from src.data.models import RetrievalResult

        return QueryTrace(
            query=query,
            retrieval_result=RetrievalResult(
                query_id=query.id,
                retrieved_passages=[],
                strategy="error",
                retrieval_time_seconds=0.0,
            ),
            agent_executions=[],
            generated_answer="",
            correctness_metrics={"exact_match": 0.0, "token_f1": 0.0, "semantic_similarity": 0.0},
            retrieval_quality_metrics={"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0},
            latency_seconds=0.0,
            cost_estimate=0.0,
            error=error_message,
        )

    def _compute_aggregate_metrics(
        self,
        query_traces: List[QueryTrace],
    ) -> AggregateMetrics:
        """Compute aggregate metrics from query traces."""
        # Extract metrics lists
        answer_metrics_list = [trace.correctness_metrics for trace in query_traces]
        retrieval_metrics_list = [trace.retrieval_quality_metrics for trace in query_traces]
        latencies = [trace.latency_seconds for trace in query_traces]
        costs = [trace.cost_estimate for trace in query_traces]

        # Use metrics calculator
        aggregated = self.metrics_calculator.compute_aggregate_metrics(
            answer_metrics_list=answer_metrics_list,
            retrieval_metrics_list=retrieval_metrics_list,
            latencies=latencies,
            costs=costs,
        )

        # Create dataclass instances
        answer_metrics = AnswerMetrics(
            exact_match_rate=aggregated["answer"]["exact_match_rate"],
            mean_token_f1=aggregated["answer"]["mean_token_f1"],
            mean_semantic_similarity=aggregated["answer"]["mean_semantic_similarity"],
        )

        retrieval_metrics = RetrievalMetrics(
            mean_precision=aggregated["retrieval"]["mean_precision"],
            mean_recall=aggregated["retrieval"]["mean_recall"],
            mean_f1=aggregated["retrieval"]["mean_f1"],
            mean_mrr=aggregated["retrieval"]["mean_mrr"],
        )

        operational_metrics = OperationalMetrics(
            mean_latency_seconds=aggregated["operational"]["mean_latency_seconds"],
            p95_latency_seconds=aggregated["operational"]["p95_latency_seconds"],
            total_tokens=int(sum([t.cost_estimate * 1000 for t in query_traces])),  # Rough estimate
            mean_cost_per_query=aggregated["operational"]["mean_cost_per_query"],
            total_cost=aggregated["operational"]["total_cost"],
        )

        return AggregateMetrics(
            total_queries=len(query_traces),
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            operational_metrics=operational_metrics,
        )

    def _print_summary(self, run: EvaluationRun):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print(f"Evaluation Complete: {run.run_id}")
        print(f"{'='*70}\n")

        if run.aggregate_metrics:
            agg = run.aggregate_metrics

            print("Answer Correctness:")
            print(f"  Exact Match Rate: {agg.answer_metrics.exact_match_rate:.3f}")
            print(f"  Mean Token F1:    {agg.answer_metrics.mean_token_f1:.3f}")
            print(f"  Semantic Sim:     {agg.answer_metrics.mean_semantic_similarity:.3f}")

            print("\nRetrieval Quality:")
            print(f"  Precision:        {agg.retrieval_metrics.mean_precision:.3f}")
            print(f"  Recall:           {agg.retrieval_metrics.mean_recall:.3f}")
            print(f"  F1:               {agg.retrieval_metrics.mean_f1:.3f}")
            print(f"  MRR:              {agg.retrieval_metrics.mean_mrr:.3f}")

            print("\nOperational:")
            print(f"  Mean Latency:     {agg.operational_metrics.mean_latency_seconds:.2f}s")
            print(f"  Total Cost:       ${agg.operational_metrics.total_cost:.2f}")
            print()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinDER RAG evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--subset",
        type=int,
        help="Evaluate only first N queries",
    )
    parser.add_argument(
        "--langfuse-enabled",
        type=bool,
        default=True,
        help="Enable Langfuse logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Run evaluation
    runner = EvaluationRunner(config, langfuse_enabled=args.langfuse_enabled)
    evaluation_run = runner.run_evaluation(dataset_subset=args.subset)

    print(f"\n✓ Evaluation run complete: {evaluation_run.run_id}")
