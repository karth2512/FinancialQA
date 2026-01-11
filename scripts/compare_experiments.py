#!/usr/bin/env python
"""
Compare multiple Langfuse experiments.

This script provides a command-line interface for comparing experiment results
and detecting regressions.

Usage:
    python scripts/compare_experiments.py run-001 run-002 run-003
    python scripts/compare_experiments.py run-001 run-002 --format markdown
    python scripts/compare_experiments.py run-002 --baseline run-001 --detect-regression
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse
from src.langfuse_integration.analysis import (
    compare_experiments,
    generate_comparison_report,
    detect_regression
)
from src.langfuse_integration.config import validate_langfuse_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Langfuse experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three experiments
  python scripts/compare_experiments.py exp-001 exp-002 exp-003

  # Compare with Markdown output
  python scripts/compare_experiments.py exp-001 exp-002 --format markdown

  # Detect regression against baseline
  python scripts/compare_experiments.py exp-002 --baseline exp-001 --detect-regression

  # Save comparison to file
  python scripts/compare_experiments.py exp-001 exp-002 --output comparison.md --format markdown
        """
    )

    parser.add_argument(
        "experiment_ids",
        nargs="+",
        help="Experiment run IDs to compare"
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline experiment ID for regression detection"
    )

    parser.add_argument(
        "--detect-regression",
        action="store_true",
        help="Detect regressions against baseline"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold as decimal (default: 0.05 = 5%%)"
    )

    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metrics to compare (if not specified, compares all)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save comparison report to file"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Validate credentials
    logger.info("Validating Langfuse credentials")
    try:
        validate_langfuse_credentials()
    except Exception as e:
        logger.error(f"Credential validation failed: {e}")
        sys.exit(1)

    # Initialize Langfuse client
    logger.info("Initializing Langfuse client")
    client = Langfuse()

    # Regression detection mode
    if args.detect_regression:
        if not args.baseline:
            logger.error("--baseline required for regression detection")
            sys.exit(1)

        if len(args.experiment_ids) != 1:
            logger.error("Regression detection requires exactly one current experiment ID")
            sys.exit(1)

        current_id = args.experiment_ids[0]

        print("\n" + "="*70)
        print("REGRESSION DETECTION")
        print("="*70)
        print(f"Baseline:  {args.baseline}")
        print(f"Current:   {current_id}")
        print(f"Threshold: {args.threshold * 100:.1f}%")
        print("="*70)
        print()

        try:
            result = detect_regression(
                client,
                current_experiment_id=current_id,
                baseline_experiment_id=args.baseline,
                threshold=args.threshold
            )

            # Display results
            if result["has_regression"]:
                print("⚠️  REGRESSIONS DETECTED")
                print()
                for reg in result["regressions"]:
                    print(f"  Metric:       {reg['metric']}")
                    print(f"  Baseline:     {reg['baseline_value']:.4f}")
                    print(f"  Current:      {reg['current_value']:.4f}")
                    print(f"  Change:       {reg['change_percent']:.2f}%")
                    print()

            if result["improvements"]:
                print("✓ IMPROVEMENTS DETECTED")
                print()
                for imp in result["improvements"]:
                    print(f"  Metric:       {imp['metric']}")
                    print(f"  Baseline:     {imp['baseline_value']:.4f}")
                    print(f"  Current:      {imp['current_value']:.4f}")
                    print(f"  Change:       +{imp['change_percent']:.2f}%")
                    print()

            if not result["has_regression"] and not result["improvements"]:
                print("✓ No significant changes detected")
                print()

            sys.exit(1 if result["has_regression"] else 0)

        except Exception as e:
            logger.error(f"Regression detection failed: {e}")
            sys.exit(1)

    # Standard comparison mode
    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    print(f"Comparing {len(args.experiment_ids)} experiments:")
    for exp_id in args.experiment_ids:
        print(f"  - {exp_id}")
    print("="*70)
    print()

    try:
        # Compare experiments
        comparison = compare_experiments(
            client,
            args.experiment_ids,
            metrics=args.metrics
        )

        # Generate report
        report = generate_comparison_report(comparison, output_format=args.format)

        # Display report
        print(report)

        # Save to file if specified
        if args.output:
            logger.info(f"Saving report to: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)

            with open(args.output, 'w') as f:
                f.write(report)

            print(f"\n✓ Report saved to: {args.output}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        print(f"\n✗ Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
