"""
Configure GraphRAG settings.yaml to use OpenAI API directly with local embeddings.

This script modifies the settings.yaml file created by `graphrag init` to:
1. Point LLM to OpenAI API using gpt-4o-mini (cheapest model)
2. Use local sentence-transformers for embeddings
3. Disable chunking (files are already pre-chunked)
4. Optimize entity extraction and community detection parameters
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_settings(settings_path: Path) -> Dict[str, Any]:
    """Load existing settings.yaml file."""
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


def save_settings(settings_path: Path, settings: Dict[str, Any]) -> None:
    """Save updated settings to YAML file."""
    with open(settings_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)


def configure_graphrag_settings(ragtest_root: str = "./ragtest") -> None:
    """
    Configure GraphRAG settings for OpenAI API and local embeddings.

    Args:
        ragtest_root: Path to GraphRAG workspace directory
    """
    ragtest_path = Path(ragtest_root)
    settings_path = ragtest_path / "settings.yaml"

    # Validate ragtest directory exists
    if not ragtest_path.exists():
        raise FileNotFoundError(
            f"GraphRAG workspace not found: {ragtest_root}\n"
            "Run 'make graphrag-init' first to initialize the workspace."
        )

    # Validate settings.yaml exists
    if not settings_path.exists():
        raise FileNotFoundError(
            f"settings.yaml not found in {ragtest_root}\n"
            "Run 'make graphrag-init' first to create the configuration."
        )

    logger.info(f"Loading settings from {settings_path}")
    settings = load_settings(settings_path)

    logger.info("Configuring LLM to use OpenAI API directly...")

    # Configure LLM to use OpenAI API directly
    if "llm" not in settings:
        settings["llm"] = {}

    settings["llm"].update({
        "api_key": "${OPENAI_API_KEY}",
        "type": "openai_chat",
        "model": "gpt-4o-mini",  # Cheapest OpenAI model
        "max_retries": 3,
        "request_timeout": 180.0,
    })

    logger.info("Configuring embeddings to use local sentence-transformers...")

    # Configure local embeddings
    if "embeddings" not in settings:
        settings["embeddings"] = {}

    settings["embeddings"].update({
        "async_mode": "threaded",
        "llm": {
            "api_key": "NONE",
            "type": "openai_embedding",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "api_base": "NONE",
        }
    })

    logger.info("Disabling chunking (files are already pre-chunked)...")

    # Disable chunking since files are already chunked
    if "chunks" not in settings:
        settings["chunks"] = {}

    settings["chunks"].update({
        "size": 99999,  # Very large to prevent chunking
        "overlap": 0,
        "group_by_columns": ["id"],
    })

    logger.info("Configuring entity extraction parameters...")

    # Configure entity extraction
    if "entity_extraction" not in settings:
        settings["entity_extraction"] = {}

    settings["entity_extraction"].update({
        "max_gleanings": 1,  # Reduce cost, still effective
        "prompt": None,  # Use default prompt
    })

    logger.info("Configuring community detection and reports...")

    # Configure community reports
    if "community_reports" not in settings:
        settings["community_reports"] = {}

    settings["community_reports"].update({
        "max_length": 1500,
        "max_input_length": 16000,
    })

    # Configure snapshot settings
    if "snapshots" not in settings:
        settings["snapshots"] = {}

    settings["snapshots"].update({
        "graphml": True,
        "raw_entities": True,
        "top_level_nodes": True,
    })

    # Save updated settings
    logger.info(f"Saving updated settings to {settings_path}")
    save_settings(settings_path, settings)

    logger.info("✓ Configuration complete!")
    logger.info("")
    logger.info("=" * 60)
    logger.info("GraphRAG Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"LLM Model: gpt-4o-mini (OpenAI, cheapest)")
    logger.info(f"LLM API: OpenAI API (direct, no proxy)")
    logger.info(f"Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)")
    logger.info(f"Chunking: Disabled (size=99999)")
    logger.info(f"Entity Extraction: max_gleanings=1")
    logger.info(f"Community Reports: max_length=1500")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Ensure OPENAI_API_KEY is set in .env")
    logger.info("2. Prepare input data: make graphrag-prepare-data")
    logger.info("3. Build the index: make graphrag-index")


def main():
    """Main entry point."""
    logger.info("Configuring GraphRAG settings...")

    try:
        configure_graphrag_settings()
    except Exception as e:
        logger.error(f"Failed to configure settings: {e}")
        raise


if __name__ == "__main__":
    main()
