"""
Prepare GraphRAG input data by copying pre-chunked text files from data/finder_text/
to the GraphRAG input directory.

The FinDER corpus text files are already optimally chunked, so we just copy them directly.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_graphrag_input(
    source_dir: str = "data/finder_text",
    target_dir: str = "./ragtest/input",
) -> Tuple[int, int]:
    """
    Copy pre-chunked text files from source to GraphRAG input directory.

    Args:
        source_dir: Directory containing pre-chunked .txt files
        target_dir: GraphRAG input directory

    Returns:
        Tuple of (num_files_copied, total_size_bytes)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Validate source directory exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Get list of .txt files
    txt_files = list(source_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {source_dir}")

    logger.info(f"Found {len(txt_files)} text files in {source_dir}")

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created target directory: {target_dir}")

    # Copy files
    total_size = 0
    copied_count = 0

    for src_file in txt_files:
        dst_file = target_path / src_file.name
        shutil.copy2(src_file, dst_file)
        total_size += dst_file.stat().st_size
        copied_count += 1

        if copied_count % 100 == 0:
            logger.info(f"Copied {copied_count}/{len(txt_files)} files...")

    logger.info(f"✓ Copied {copied_count} files")
    logger.info(f"✓ Total size: {total_size / 1024 / 1024:.2f} MB")
    logger.info(f"✓ Average file size: {total_size / copied_count / 1024:.2f} KB")

    return copied_count, total_size


def main():
    """Main entry point."""
    logger.info("Preparing GraphRAG input data...")
    logger.info("Note: Files are already pre-chunked, no additional processing needed")

    try:
        num_files, total_size = prepare_graphrag_input()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Data preparation complete!")
        logger.info(f"Files copied: {num_files}")
        logger.info(f"Total size: {total_size / 1024 / 1024:.2f} MB")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Ensure LiteLLM proxy is running: make start-litellm")
        logger.info("2. Configure GraphRAG settings: make graphrag-configure")
        logger.info("3. Build the index: make graphrag-index")

    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise


if __name__ == "__main__":
    main()
