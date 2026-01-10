"""
Langfuse integration for logging evaluation runs and query traces.

This module handles all interactions with Langfuse for observability and tracking.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.evaluation.models import EvaluationRun, QueryTrace


class LangfuseLogger:
    """Logger for evaluation results to Langfuse."""

    def __init__(self, enabled: bool = True):
        """
        Initialize Langfuse logger.

        Args:
            enabled: Whether to actually log to Langfuse (can disable for testing)
        """
        self.enabled = enabled
        self.langfuse = None

        if self.enabled:
            try:
                from langfuse import Langfuse

                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

                if not public_key or not secret_key:
                    print("Warning: Langfuse credentials not found. Logging disabled.")
                    print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.")
                    self.enabled = False
                    return

                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )

                print(f"✓ Langfuse logger initialized (host: {host})")

            except ImportError:
                print("Warning: langfuse package not installed. Logging disabled.")
                print("Install with: pip install langfuse")
                self.enabled = False
            except Exception as e:
                print(f"Warning: Failed to initialize Langfuse: {e}")
                self.enabled = False

    def log_evaluation_run(
        self,
        run: EvaluationRun,
    ) -> Optional[str]:
        """
        Log complete evaluation run as top-level trace.

        Args:
            run: Evaluation run to log

        Returns:
            Langfuse trace ID if successful, None otherwise
        """
        if not self.enabled or self.langfuse is None:
            return None

        try:
            # Create top-level trace
            trace = self.langfuse.trace(
                name=f"evaluation_run_{run.run_id}",
                metadata={
                    "run_id": run.run_id,
                    "version": run.version,
                    "timestamp": run.timestamp,
                    "dataset_version": run.dataset_version,
                    "pipeline_type": run.config.get("pipeline_type", "unknown"),
                    "status": run.status,
                },
                tags=[run.run_id, run.status],
            )

            # Log aggregate metrics if available
            if run.aggregate_metrics:
                trace.update(
                    output={
                        "total_queries": run.aggregate_metrics.total_queries,
                        "answer_metrics": {
                            "exact_match_rate": run.aggregate_metrics.answer_metrics.exact_match_rate,
                            "mean_token_f1": run.aggregate_metrics.answer_metrics.mean_token_f1,
                            "mean_semantic_similarity": run.aggregate_metrics.answer_metrics.mean_semantic_similarity,
                        },
                        "retrieval_metrics": {
                            "mean_precision": run.aggregate_metrics.retrieval_metrics.mean_precision,
                            "mean_recall": run.aggregate_metrics.retrieval_metrics.mean_recall,
                            "mean_f1": run.aggregate_metrics.retrieval_metrics.mean_f1,
                        },
                        "operational_metrics": {
                            "mean_latency_seconds": run.aggregate_metrics.operational_metrics.mean_latency_seconds,
                            "total_cost": run.aggregate_metrics.operational_metrics.total_cost,
                        },
                    }
                )

            # Log individual query traces
            for query_trace in run.query_traces:
                self.log_query_trace(query_trace, trace.id)

            # Flush to ensure data is sent
            self.langfuse.flush()

            print(f"✓ Logged evaluation run to Langfuse: {run.run_id}")
            return trace.id

        except Exception as e:
            print(f"Warning: Failed to log evaluation run to Langfuse: {e}")
            return None

    def log_query_trace(
        self,
        trace: QueryTrace,
        parent_trace_id: str,
    ) -> Optional[str]:
        """
        Log individual query execution as nested span.

        Args:
            trace: Query trace to log
            parent_trace_id: Parent evaluation run trace ID

        Returns:
            Langfuse span ID if successful, None otherwise
        """
        if not self.enabled or self.langfuse is None:
            return None

        try:
            # Create span for this query
            span = self.langfuse.span(
                name=f"query_{trace.query.id}",
                trace_id=parent_trace_id,
                input={
                    "query_id": trace.query.id,
                    "query_text": trace.query.text,
                    "expected_answer": trace.query.expected_answer,
                },
                output={
                    "generated_answer": trace.generated_answer,
                    "correctness_metrics": trace.correctness_metrics,
                    "retrieval_quality_metrics": trace.retrieval_quality_metrics,
                },
                metadata={
                    "latency_seconds": trace.latency_seconds,
                    "cost_estimate": trace.cost_estimate,
                    "error": trace.error,
                    "query_metadata": {
                        "domain_term_count": trace.query.metadata.domain_term_count,
                        "has_ambiguity": trace.query.metadata.has_ambiguity,
                        "query_type": trace.query.metadata.query_type,
                    },
                },
            )

            # Log retrieval span
            if trace.retrieval_result:
                self.langfuse.span(
                    name="retrieval",
                    trace_id=parent_trace_id,
                    parent_span_id=span.id,
                    input={"query_text": trace.query.text},
                    output={
                        "strategy": trace.retrieval_result.strategy,
                        "num_passages": len(trace.retrieval_result.retrieved_passages),
                        "retrieval_time": trace.retrieval_result.retrieval_time_seconds,
                    },
                    metadata={
                        "top_passages": [
                            {
                                "rank": p.rank,
                                "score": p.score,
                                "text_preview": p.passage.text[:100],
                            }
                            for p in trace.retrieval_result.retrieved_passages[:3]
                        ]
                    },
                )

            # Log agent executions
            for agent_exec in trace.agent_executions:
                self.langfuse.span(
                    name=f"agent_{agent_exec.input_context.get('agent_type', 'unknown')}",
                    trace_id=parent_trace_id,
                    parent_span_id=span.id,
                    input=agent_exec.input_context,
                    output=agent_exec.output_context,
                    metadata={
                        "decisions": agent_exec.decisions,
                        "latency_seconds": agent_exec.latency_seconds,
                        "cost_estimate": agent_exec.cost_estimate,
                    },
                )

            return span.id

        except Exception as e:
            print(f"Warning: Failed to log query trace to Langfuse: {e}")
            return None

    def query_evaluation_history(
        self,
        limit: int = 10,
        run_id_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical evaluation runs from Langfuse.

        Args:
            limit: Maximum number of runs to retrieve
            run_id_pattern: Filter by run_id pattern (e.g., "v*", "exp-*")

        Returns:
            List of evaluation run summaries

        Note: This is a placeholder - actual implementation would query Langfuse API
        """
        if not self.enabled or self.langfuse is None:
            return []

        try:
            # TODO: Implement actual Langfuse query
            # This would use Langfuse's API to fetch historical traces
            print("Note: query_evaluation_history not yet implemented")
            return []

        except Exception as e:
            print(f"Warning: Failed to query evaluation history: {e}")
            return []
