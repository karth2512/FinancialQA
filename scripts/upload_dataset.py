#!/usr/bin/env python
"""
Upload FinDER dataset to Langfuse with progress reporting.

This script provides a command-line interface for uploading the FinDER dataset
to Langfuse with various filtering options and configuration.

Usage:
    python scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all
    python scripts/upload_dataset.py --name financial_qa_reasoning --filter reasoning_required
    python scripts/upload_dataset.py --name financial_qa_equity --filter by_category --category equity
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse
from src.langfuse_integration.dataset_manager import upload_finder_dataset
from src.langfuse_integration.models import LangfuseDatasetConfig, DatasetFilterType
from src.langfuse_integration.config import validate_langfuse_credentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_finder_queries() -> List[Any]:
    """
    Load FinDER queries from local data directory.

    Returns:
        List of Query objects

    Raises:
        FileNotFoundError: If FinDER data not found
    """
    try:
        from src.data_handler.loader import FinDERLoader
        loader = FinDERLoader()
        queries = loader.load()
        logger.info(f"Loaded {len(queries)} queries from FinDER dataset")
        return queries
    except ImportError as e:
        logger.error(f"FinDER data loader import failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"FinDER dataset not found: {e}")
        logger.error("Run 'make download-data' to download the dataset first")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load FinDER dataset: {e}")
        logger.error("Run 'make download-data' to download the dataset first")
        sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload FinDER dataset to Langfuse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload full dataset
  python scripts/upload_dataset.py --name financial_qa_benchmark_v1 --filter all

  # Upload reasoning-only queries
  python scripts/upload_dataset.py --name financial_qa_reasoning --filter reasoning_required

  # Upload by category
  python scripts/upload_dataset.py --name financial_qa_equity --filter by_category --category equity

  # Upload with item limit (testing)
  python scripts/upload_dataset.py --name financial_qa_test --filter all --max-items 100
        """
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name in Langfuse (e.g., 'financial_qa_benchmark_v1')"
    )

    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Dataset description (default: auto-generated based on filter)"
    )

    parser.add_argument(
        "--filter",
        type=str,
        choices=["all", "reasoning_required", "no_reasoning", "by_category", "by_query_type", "by_complexity"],
        default="all",
        help="Filtering criteria for dataset subset"
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Financial category (required for --filter by_category)"
    )

    parser.add_argument(
        "--query-type",
        type=str,
        help="Query type (required for --filter by_query_type)"
    )

    parser.add_argument(
        "--min-complexity",
        type=int,
        help="Minimum complexity level (for --filter by_complexity)"
    )

    parser.add_argument(
        "--max-complexity",
        type=int,
        help="Maximum complexity level (for --filter by_complexity)"
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version tag for dataset tracking"
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to upload (for testing)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of items per batch (default: 50)"
    )

    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between batches (default: 0.5)"
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude FinDER metadata from dataset items"
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and proceed with upload"
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
        logger.error("Please check your .env file and ensure LANGFUSE_PUBLIC_KEY, "
                    "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are set correctly")
        sys.exit(1)

    # Initialize Langfuse client
    logger.info("Initializing Langfuse client")
    client = Langfuse()

    # Load FinDER queries
    logger.info("Loading FinDER dataset")
    queries = load_finder_queries()

    # Build filter params
    filter_params = None
    if args.filter == "by_category":
        if not args.category:
            logger.error("--category required when --filter is by_category")
            sys.exit(1)
        filter_params = {"category": args.category}

    elif args.filter == "by_query_type":
        if not args.query_type:
            logger.error("--query-type required when --filter is by_query_type")
            sys.exit(1)
        filter_params = {"query_type": args.query_type}

    elif args.filter == "by_complexity":
        filter_params = {}
        if args.min_complexity is not None:
            filter_params["min_complexity"] = args.min_complexity
        if args.max_complexity is not None:
            filter_params["max_complexity"] = args.max_complexity

        if not filter_params:
            logger.error("--min-complexity or --max-complexity required when --filter is by_complexity")
            sys.exit(1)

    # Auto-generate description if not provided
    description = args.description
    if not description:
        if args.filter == "all":
            description = f"Complete FinDER dataset for RAG evaluation ({len(queries)} queries)"
        elif args.filter == "reasoning_required":
            description = "FinDER queries requiring multi-hop reasoning"
        elif args.filter == "no_reasoning":
            description = "FinDER queries without reasoning requirements"
        elif args.filter == "by_category":
            description = f"FinDER queries in category: {args.category}"
        elif args.filter == "by_query_type":
            description = f"FinDER queries of type: {args.query_type}"
        else:
            description = f"FinDER dataset with filter: {args.filter}"

    # Create dataset config
    config = LangfuseDatasetConfig(
        name=args.name,
        description=description,
        filter_type=DatasetFilterType(args.filter),
        filter_params=filter_params,
        version_tag=args.version,
        include_metadata=not args.no_metadata,
        metadata_enrichment={
            "source": "FinDER",
            "uploaded_by": "upload_dataset.py",
            "filter_type": args.filter
        },
        max_items=args.max_items
    )

    # Display configuration summary
    print("\n" + "="*60)
    print("DATASET UPLOAD CONFIGURATION")
    print("="*60)
    print(f"Dataset Name:       {config.name}")
    print(f"Description:        {config.description}")
    print(f"Filter Type:        {config.filter_type}")
    if filter_params:
        print(f"Filter Params:      {filter_params}")
    if args.version:
        print(f"Version Tag:        {args.version}")
    if args.max_items:
        print(f"Max Items:          {args.max_items}")
    print(f"Include Metadata:   {config.include_metadata}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Rate Limit Delay:   {args.rate_limit_delay}s")
    print("="*60)
    print()

    # Confirm upload (skip if --yes flag is provided)
    if not args.yes:
        print("Ready to upload. Continue? (yes/no): ", end="")
        response = input().strip().lower()
        if response not in ["yes", "y"]:
            print("Upload cancelled.")
            sys.exit(0)

    # Upload dataset
    try:
        print("\nUploading dataset to Langfuse...")
        result = upload_finder_dataset(
            client=client,
            config=config,
            queries=queries,
            batch_size=args.batch_size,
            rate_limit_delay=args.rate_limit_delay
        )

        # Display results
        print("\n" + "="*60)
        print("UPLOAD SUMMARY")
        print("="*60)
        print(f"Dataset Name:       {result['dataset_name']}")
        print(f"Total Items:        {result['total_items']}")
        print(f"Successful:         {result['successful_items']}")
        print(f"Failed:             {result['failed_items']}")
        print(f"Upload Time:        {result['upload_time_seconds']:.2f}s")

        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(result['errors']) > 10:
                print(f"  ... and {len(result['errors']) - 10} more errors")

        print("="*60)
        print()

        # Success/failure status
        if result['failed_items'] == 0:
            print("✓ Upload completed successfully!")
            print(f"\nView your dataset in Langfuse UI:")
            print(f"  https://cloud.langfuse.com (or your self-hosted URL)")
            sys.exit(0)
        elif result['successful_items'] > 0:
            print(f"⚠ Upload completed with {result['failed_items']} failures")
            sys.exit(1)
        else:
            print("✗ Upload failed completely")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nUpload interrupted by user")
        client.flush()
        sys.exit(1)

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        print(f"\n✗ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
