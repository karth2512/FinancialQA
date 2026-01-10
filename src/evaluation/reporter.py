"""
Report generator for evaluation summaries, comparisons, and failure analysis.

This module generates human-readable reports from evaluation results.
"""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from src.evaluation.models import EvaluationRun, QueryTrace


class ReportGenerator:
    """Generator for evaluation reports in markdown format."""

    def generate_summary_report(
        self,
        run: EvaluationRun,
        output_format: str = "markdown",
    ) -> str:
        """
        Generate summary report for an evaluation run.

        Args:
            run: Evaluation run to summarize
            output_format: "markdown" or "json"

        Returns:
            Report content as string
        """
        if output_format == "json":
            return self._generate_json_summary(run)

        # Markdown format
        report = [
            f"# Evaluation Report: {run.run_id}",
            "",
            f"**Run ID**: {run.run_id}",
            f"**Version**: {run.version or 'N/A'}",
            f"**Timestamp**: {run.timestamp}",
            f"**Dataset**: FinDER {run.dataset_version}",
            f"**Status**: {run.status}",
            "",
        ]

        if run.aggregate_metrics:
            agg = run.aggregate_metrics

            report.extend([
                "## Aggregate Metrics",
                "",
                "### Answer Correctness",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Exact Match Rate | {agg.answer_metrics.exact_match_rate:.3f} |",
                f"| Mean Token F1 | {agg.answer_metrics.mean_token_f1:.3f} |",
                f"| Mean Semantic Similarity | {agg.answer_metrics.mean_semantic_similarity:.3f} |",
                "",
                "### Retrieval Quality",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean Precision | {agg.retrieval_metrics.mean_precision:.3f} |",
                f"| Mean Recall | {agg.retrieval_metrics.mean_recall:.3f} |",
                f"| Mean F1 | {agg.retrieval_metrics.mean_f1:.3f} |",
                f"| Mean MRR | {agg.retrieval_metrics.mean_mrr:.3f} |",
                "",
                "### Operational Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean Latency | {agg.operational_metrics.mean_latency_seconds:.2f}s |",
                f"| P95 Latency | {agg.operational_metrics.p95_latency_seconds:.2f}s |",
                f"| Total Tokens | {agg.operational_metrics.total_tokens:,} |",
                f"| Mean Cost/Query | ${agg.operational_metrics.mean_cost_per_query:.4f} |",
                f"| Total Cost | ${agg.operational_metrics.total_cost:.2f} |",
                "",
            ])

            # Performance breakdowns
            if agg.breakdown_by_complexity:
                report.extend([
                    "## Performance by Query Complexity",
                    "",
                    "| Domain Terms | Queries | Accuracy | F1 |",
                    "|-------------|---------|----------|-----|",
                ])
                for complexity, metrics in agg.breakdown_by_complexity.items():
                    report.append(
                        f"| {complexity} | {metrics.get('count', 'N/A')} | "
                        f"{metrics.get('accuracy', 0):.3f} | {metrics.get('f1', 0):.3f} |"
                    )
                report.append("")

        report.extend([
            "---",
            f"*Generated at {datetime.utcnow().isoformat()}*",
        ])

        return "\n".join(report)

    def generate_comparison_report(
        self,
        run1: EvaluationRun,
        run2: EvaluationRun,
        output_format: str = "markdown",
    ) -> str:
        """
        Compare two evaluation runs.

        Args:
            run1: First evaluation run (baseline)
            run2: Second evaluation run (comparison)
            output_format: "markdown" or "json"

        Returns:
            Comparison report showing metric deltas
        """
        if not run1.aggregate_metrics or not run2.aggregate_metrics:
            return "Error: Both runs must have aggregate metrics for comparison"

        agg1 = run1.aggregate_metrics
        agg2 = run2.aggregate_metrics

        report = [
            f"# Comparison Report: {run1.run_id} vs {run2.run_id}",
            "",
            f"**Baseline**: {run1.run_id} ({run1.timestamp})",
            f"**Comparison**: {run2.run_id} ({run2.timestamp})",
            "",
            "## Metric Changes",
            "",
            "| Metric | Baseline | Comparison | Delta | % Change |",
            "|--------|----------|------------|-------|----------|",
        ]

        # Answer metrics
        em1 = agg1.answer_metrics.exact_match_rate
        em2 = agg2.answer_metrics.exact_match_rate
        em_delta = em2 - em1
        em_pct = (em_delta / em1 * 100) if em1 > 0 else 0

        f1_1 = agg1.answer_metrics.mean_token_f1
        f1_2 = agg2.answer_metrics.mean_token_f1
        f1_delta = f1_2 - f1_1
        f1_pct = (f1_delta / f1_1 * 100) if f1_1 > 0 else 0

        # Retrieval metrics
        ret_f1_1 = agg1.retrieval_metrics.mean_f1
        ret_f1_2 = agg2.retrieval_metrics.mean_f1
        ret_f1_delta = ret_f1_2 - ret_f1_1
        ret_f1_pct = (ret_f1_delta / ret_f1_1 * 100) if ret_f1_1 > 0 else 0

        # Operational metrics
        lat1 = agg1.operational_metrics.mean_latency_seconds
        lat2 = agg2.operational_metrics.mean_latency_seconds
        lat_delta = lat2 - lat1
        lat_pct = (lat_delta / lat1 * 100) if lat1 > 0 else 0

        cost1 = agg1.operational_metrics.mean_cost_per_query
        cost2 = agg2.operational_metrics.mean_cost_per_query
        cost_delta = cost2 - cost1
        cost_pct = (cost_delta / cost1 * 100) if cost1 > 0 else 0

        report.extend([
            f"| Exact Match | {em1:.3f} | {em2:.3f} | {em_delta:+.3f} | {em_pct:+.1f}% |",
            f"| Token F1 | {f1_1:.3f} | {f1_2:.3f} | {f1_delta:+.3f} | {f1_pct:+.1f}% |",
            f"| Retrieval F1 | {ret_f1_1:.3f} | {ret_f1_2:.3f} | {ret_f1_delta:+.3f} | {ret_f1_pct:+.1f}% |",
            f"| Latency (s) | {lat1:.2f} | {lat2:.2f} | {lat_delta:+.2f} | {lat_pct:+.1f}% |",
            f"| Cost/Query ($) | ${cost1:.4f} | ${cost2:.4f} | ${cost_delta:+.4f} | {cost_pct:+.1f}% |",
            "",
            "## Conclusion",
            "",
        ])

        # Generate conclusion
        improvements = []
        regressions = []

        if f1_delta > 0.05:
            improvements.append(f"Answer F1 improved by {f1_pct:.1f}%")
        elif f1_delta < -0.05:
            regressions.append(f"Answer F1 decreased by {abs(f1_pct):.1f}%")

        if ret_f1_delta > 0.05:
            improvements.append(f"Retrieval F1 improved by {ret_f1_pct:.1f}%")

        if lat_pct < -10:
            improvements.append(f"Latency reduced by {abs(lat_pct):.1f}%")
        elif lat_pct > 20:
            regressions.append(f"Latency increased by {lat_pct:.1f}%")

        if improvements:
            report.append("**Improvements:**")
            for imp in improvements:
                report.append(f"- {imp}")
            report.append("")

        if regressions:
            report.append("**Regressions:**")
            for reg in regressions:
                report.append(f"- {reg}")
            report.append("")

        report.extend([
            "---",
            f"*Generated at {datetime.utcnow().isoformat()}*",
        ])

        return "\n".join(report)

    def generate_failure_analysis(
        self,
        run: EvaluationRun,
        failure_threshold: float = 0.5,
    ) -> str:
        """
        Analyze failed queries and categorize root causes.

        Args:
            run: Evaluation run to analyze
            failure_threshold: Token F1 below this is considered failure

        Returns:
            Failure analysis report
        """
        failures = [
            trace for trace in run.query_traces
            if trace.correctness_metrics.get("token_f1", 1.0) < failure_threshold
        ]

        total_failures = len(failures)
        total_queries = len(run.query_traces)
        failure_rate = total_failures / total_queries if total_queries > 0 else 0

        report = [
            f"# Failure Analysis: {run.run_id}",
            "",
            f"**Total Failures**: {total_failures} / {total_queries} ({failure_rate:.1%})",
            f"**Failure Threshold**: Token F1 < {failure_threshold}",
            "",
            "## Failure Categories",
            "",
        ]

        # Categorize failures (simplified categorization)
        categories = {
            "Low Retrieval Quality": [],
            "Ambiguous Query": [],
            "Complex Query": [],
            "Other": [],
        }

        for failure in failures:
            # Categorize based on available metrics
            if failure.retrieval_quality_metrics.get("f1", 1.0) < 0.3:
                categories["Low Retrieval Quality"].append(failure)
            elif failure.query.metadata.has_ambiguity:
                categories["Ambiguous Query"].append(failure)
            elif failure.query.metadata.domain_term_count >= 5:
                categories["Complex Query"].append(failure)
            else:
                categories["Other"].append(failure)

        report.extend([
            "| Category | Count | % of Failures |",
            "|----------|-------|---------------|",
        ])

        for category, failures_list in categories.items():
            count = len(failures_list)
            pct = count / total_failures * 100 if total_failures > 0 else 0
            report.append(f"| {category} | {count} | {pct:.1f}% |")

        report.append("")

        # Show examples
        for category, failures_list in categories.items():
            if failures_list:
                example = failures_list[0]
                report.extend([
                    f"### Example: {category}",
                    "",
                    f"**Query**: {example.query.text}",
                    f"**Expected Answer**: {example.query.expected_answer}",
                    f"**Generated Answer**: {example.generated_answer}",
                    f"**Token F1**: {example.correctness_metrics.get('token_f1', 0):.3f}",
                    f"**Retrieval F1**: {example.retrieval_quality_metrics.get('f1', 0):.3f}",
                    "",
                ])

        return "\n".join(report)

    def _generate_json_summary(self, run: EvaluationRun) -> str:
        """Generate JSON format summary."""
        import json

        summary = {
            "run_id": run.run_id,
            "version": run.version,
            "timestamp": run.timestamp,
            "dataset_version": run.dataset_version,
            "status": run.status,
        }

        if run.aggregate_metrics:
            summary["metrics"] = {
                "answer": {
                    "exact_match_rate": run.aggregate_metrics.answer_metrics.exact_match_rate,
                    "mean_token_f1": run.aggregate_metrics.answer_metrics.mean_token_f1,
                },
                "retrieval": {
                    "mean_precision": run.aggregate_metrics.retrieval_metrics.mean_precision,
                    "mean_recall": run.aggregate_metrics.retrieval_metrics.mean_recall,
                    "mean_f1": run.aggregate_metrics.retrieval_metrics.mean_f1,
                },
                "operational": {
                    "mean_latency_seconds": run.aggregate_metrics.operational_metrics.mean_latency_seconds,
                    "total_cost": run.aggregate_metrics.operational_metrics.total_cost,
                },
            }

        return json.dumps(summary, indent=2)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--run1", help="First run ID for comparison")
    parser.add_argument("--run2", help="Second run ID for comparison")
    parser.add_argument("--failure-analysis", action="store_true", help="Generate failure analysis")
    parser.add_argument("--output", type=Path, help="Output file path")

    args = parser.parse_args()

    # TODO: Implement CLI interface with actual run loading
    print("Report generator CLI - to be implemented with run loading")
