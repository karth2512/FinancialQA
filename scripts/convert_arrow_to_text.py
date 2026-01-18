"""
Convert FinDER Arrow dataset to individual text files.

This script converts the FinDER dataset from Arrow format to individual text files,
with one .txt file per query containing metadata fields in pipe-delimited format.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.ipc as ipc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for filesystems.

    Args:
        filename: Original filename string

    Returns:
        Sanitized filename safe for Windows and Unix filesystems
    """
    # Replace special characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    # Limit length to 255 characters (filesystem limit)
    sanitized = sanitized[:255]
    return sanitized


def find_arrow_file(input_dir: Path) -> Optional[Path]:
    """
    Find the Arrow file in the input directory.

    Args:
        input_dir: Root directory containing Arrow files

    Returns:
        Path to the Arrow file, or None if not found
    """
    arrow_files = list(input_dir.rglob("*.arrow"))
    if not arrow_files:
        return None
    if len(arrow_files) > 1:
        logger.warning(f"Found {len(arrow_files)} Arrow files, using first: {arrow_files[0]}")
    return arrow_files[0]


def convert_arrow_to_text(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Convert Arrow dataset to individual text files.

    Args:
        input_dir: Directory containing Arrow files
        output_dir: Directory to write text files
        overwrite: Whether to overwrite existing files
    """
    # Find Arrow file
    logger.info(f"Searching for Arrow files in {input_dir}")
    arrow_file = find_arrow_file(input_dir)

    if arrow_file is None:
        logger.error(f"No Arrow files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found Arrow file: {arrow_file}")

    # Load Arrow file
    logger.info("Loading Arrow file...")
    try:
        # Try streaming format first (HuggingFace datasets use this)
        with ipc.open_stream(arrow_file) as reader:
            table = reader.read_all()
    except Exception as stream_error:
        # Fallback to file format
        try:
            with ipc.open_file(arrow_file) as reader:
                table = reader.read_all()
        except Exception as file_error:
            logger.error(f"Failed to load Arrow file (tried both stream and file formats)")
            logger.error(f"Stream error: {stream_error}")
            logger.error(f"File error: {file_error}")
            sys.exit(1)

    # Convert to pandas for easier iteration
    df = table.to_pandas()
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} queries from Arrow file")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each query
    created_count = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        try:
            # Generate filename from _id field
            query_id = str(row.get("_id", f"query_{idx}"))
            filename = sanitize_filename(query_id) + ".txt"
            filepath = output_dir / filename

            # Skip if file exists and not overwriting
            if filepath.exists() and not overwrite:
                skipped_count += 1
                if skipped_count % 500 == 0:
                    logger.info(f"Progress: {created_count + skipped_count}/{total_rows} (skipped {skipped_count})")
                continue

            # Extract fields
            query_id = str(row.get("_id", "N/A"))
            category = str(row.get("category", "N/A"))
            type_field = str(row.get("type", "N/A"))

            # Handle references (list field) - take first reference
            references = row.get("references", [])
            # Check if references is not None and has length before accessing
            if references is not None and hasattr(references, '__len__') and len(references) > 0:
                reference = str(references[0])
            else:
                reference = "N/A"

            # Format text content with pipe delimiters
            content = (
                f"ID: {query_id}|\n"
                f"CATEGORY: {category}|\n"
                f"TYPE: {type_field}|\n"
                f"REFERENCE: {reference}|\n"
            )

            # Write to file
            filepath.write_text(content, encoding="utf-8")
            created_count += 1

            # Log progress every 500 files
            if created_count % 500 == 0:
                logger.info(f"Progress: {created_count}/{total_rows} files created")

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing row {idx}: {e}")
            continue

    # Final summary
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"Total queries: {total_rows}")
    logger.info(f"Files created: {created_count}")
    logger.info(f"Files skipped: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert FinDER Arrow dataset to individual text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/finder"),
        help="Path to finder data directory (default: data/finder)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finder_text"),
        help="Path to output text files directory (default: data/finder_text)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: False)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Run conversion
    convert_arrow_to_text(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
