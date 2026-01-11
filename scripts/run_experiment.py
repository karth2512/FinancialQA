#!/usr/bin/env python
"""
Run Langfuse experiment from configuration file.

This script provides a command-line interface for executing RAG experiments
with Langfuse tracing and evaluation.

Usage:
    python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml
    python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse
from src.langfuse_integration.experiment_runner import run_langfuse_experiment
from src.config.experiments import LangfuseExperimentConfig
from src.langfuse_integration.config import validate_langfuse_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Langfuse experiment from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiment
  python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml

  # Dry run (validate config without executing)
  python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --dry-run

  # Override concurrency
  python scripts/run_experiment.py experiments/configs/langfuse_baseline.yaml --concurrency 5
        """,
    )

    parser.add_argument(
        "config_path", type=Path, help="Path to experiment configuration YAML file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiment",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override max_concurrency from config",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of dataset items to process (for testing)",
    )

    parser.add_argument(
        "--disable-evaluators",
        action="store_true",
        help="Disable all evaluators (faster execution for testing)",
    )

    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save experiment results JSON"
    )

    return parser.parse_args()


def display_config_summary(config: LangfuseExperimentConfig):
    """Display experiment configuration summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Name:                {config.name}")
    print(f"Run ID:              {config.run_id}")
    print(f"Description:         {config.description}")
    print(f"Pipeline Type:       {config.pipeline_type}")
    print()
    print("Retrieval Configuration:")
    print(f"  Strategy:          {config.retrieval_config.strategy}")
    print(f"  Top-K:             {config.retrieval_config.top_k}")
    if config.retrieval_config.reranking:
        print(f"  Reranking:         Enabled")
    print()
    print("LLM Configuration:")
    for agent, llm_config in config.llm_configs.items():
        print(f"  {agent}:")
        print(f"    Provider:        {llm_config.provider}")
        print(f"    Model:           {llm_config.model}")
        print(f"    Temperature:     {llm_config.temperature}")
        print(f"    Max Tokens:      {llm_config.max_tokens}")
    print()
    print("Dataset Configuration:")
    if config.langfuse_dataset_name:
        print(f"  Dataset:           {config.langfuse_dataset_name} (Langfuse)")
    elif config.use_local_data:
        print(f"  Dataset:           Local data")
        print(f"  Path:              {config.local_data_path}")
    print()
    print("Execution Configuration:")
    print(f"  Max Concurrency:   {config.max_concurrency}")
    print(f"  Flush Interval:    {config.flush_interval}s")
    print(f"  Flush Batch Size:  {config.flush_batch_size}")
    print()
    print("Evaluation Configuration:")
    print(
        f"  Item Evaluators:   {', '.join(config.item_evaluator_names) if config.enable_item_evaluators else 'Disabled'}"
    )
    print(
        f"  Run Evaluators:    {', '.join(config.run_evaluator_names) if config.enable_run_evaluators else 'Disabled'}"
    )
    if config.evaluation_thresholds:
        print(f"  Thresholds:        {config.evaluation_thresholds}")
    print()
    print("Tags and Metadata:")
    print(f"  Tags:              {', '.join(config.langfuse_tags)}")
    if config.langfuse_metadata:
        print(f"  Metadata:          {config.langfuse_metadata}")
    print("=" * 70)
    print()


def main():
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Validate config file exists
    if not args.config_path.exists():
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config_path}")
    try:
        config = LangfuseExperimentConfig.from_yaml(args.config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Apply command-line overrides
    if args.concurrency is not None:
        config.max_concurrency = args.concurrency
        logger.info(f"Overriding max_concurrency to {args.concurrency}")

    if args.disable_evaluators:
        config.enable_item_evaluators = False
        config.enable_run_evaluators = False
        logger.info("Disabled all evaluators")

    # Display configuration
    display_config_summary(config)

    # Dry run mode
    if args.dry_run:
        print("[OK] Configuration is valid")
        print("\nDry run complete. Use without --dry-run to execute experiment.")
        sys.exit(0)

    # Validate Langfuse credentials
    logger.info("Validating Langfuse credentials")
    try:
        validate_langfuse_credentials()
    except Exception as e:
        logger.error(f"Credential validation failed: {e}")
        logger.error("Please check your .env file")
        sys.exit(1)

    # Estimate costs and get user confirmation
    print("WARNING:  WARNING: This experiment will incur costs from:")
    generator_config = config.llm_configs.get("generator")
    if generator_config:
        provider = generator_config.provider
        model = generator_config.model
        print(f"   - {provider.upper()} API calls ({model})")
    else:
        print("   - LLM API calls")
    print("   - Langfuse API usage")
    print()

    if config.langfuse_dataset_name:
        print(f"Dataset: {config.langfuse_dataset_name}")
        print("Note: Exact item count will be determined when dataset is loaded")

    print(f"Concurrency: {config.max_concurrency}")
    print()
    print("Continue? (yes/no): ", end="")
    response = "y"
    if response not in ["yes", "y"]:
        print("Experiment cancelled.")
        sys.exit(0)

    # Initialize Langfuse client
    logger.info("Initializing Langfuse client")
    client = Langfuse(
        flush_at=config.flush_batch_size, flush_interval=config.flush_interval
    )

    # Run experiment
    try:
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT")
        print("=" * 70)
        print()

        result = run_langfuse_experiment(config, client)

        # Display results
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"Experiment:          {result.experiment_name}")
        print(f"Run ID:              {result.run_id}")
        print(f"Status:              {result.status.upper()}")
        print()
        print(f"Total Items:         {result.total_items}")
        print(f"Successful:          {result.successful_items}")
        print(f"Failed:              {result.failed_items}")
        print(f"Success Rate:        {result.get_success_rate():.2%}")
        print()
        print(f"Total Time:          {result.total_latency_seconds:.2f}s")
        print(f"Average Latency:     {result.get_average_latency():.2f}s per item")
        print()

        if result.aggregate_scores:
            print("Aggregate Scores:")
            for metric, score in result.aggregate_scores.items():
                print(f"  {metric:30s} {score:.4f}")
            print()

        print(f"Langfuse URL:        {result.langfuse_url}")
        print("=" * 70)
        print()

        # Save results if output path specified
        if args.output:
            import json

            logger.info(f"Saving results to: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)

            with open(args.output, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)

            print(f"[OK] Results saved to: {args.output}")

        # Success status
        if result.status == "completed":
            print("[OK] Experiment completed successfully!")
            sys.exit(0)
        elif result.status == "partial":
            print(f" Experiment completed with {result.failed_items} failures")
            sys.exit(1)
        else:
            print("[ERROR] Experiment failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        client.flush()
        sys.exit(1)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\n[ERROR] Experiment failed: {e}")
        sys.exit(1)

    finally:
        # Ensure client is flushed
        logger.info("Flushing Langfuse client")
        client.flush()
        client.shutdown()


if __name__ == "__main__":
    main()
