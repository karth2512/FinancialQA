"""
Dataset management for Langfuse integration.

This module provides functionality for creating, uploading, and managing
FinDER datasets in Langfuse with versioning, filtering, and metadata enrichment.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from langfuse import Langfuse

from .models import LangfuseDatasetConfig, DatasetItemMapping, DatasetFilterType
from .retry import retry_with_backoff
from .exceptions import DatasetUploadError, DatasetNotFoundError

logger = logging.getLogger(__name__)


def create_dataset(client: Langfuse, config: LangfuseDatasetConfig) -> Dict[str, Any]:
    """
    Create a new dataset in Langfuse.

    Args:
        client: Initialized Langfuse client
        config: Dataset configuration

    Returns:
        Dict with dataset creation info (name, description, metadata)

    Raises:
        DatasetUploadError: If dataset creation fails
    """
    try:
        logger.info(f"Creating dataset: {config.name}")

        # Create dataset with metadata
        dataset = client.create_dataset(
            name=config.name,
            description=config.description,
            metadata=config.metadata_enrichment if config.metadata_enrichment else {},
        )

        logger.info(f"Dataset '{config.name}' created successfully")

        return {
            "name": config.name,
            "description": config.description,
            "metadata": config.metadata_enrichment or {},
        }

    except Exception as e:
        logger.error(f"Failed to create dataset '{config.name}': {e}")
        raise DatasetUploadError(f"Dataset creation failed: {e}") from e


def get_dataset(client: Langfuse, dataset_name: str) -> Any:
    """
    Retrieve an existing dataset from Langfuse.

    Args:
        client: Initialized Langfuse client
        dataset_name: Name of the dataset to retrieve

    Returns:
        Dataset object from Langfuse

    Raises:
        DatasetNotFoundError: If dataset doesn't exist
    """
    try:
        logger.info(f"Retrieving dataset: {dataset_name}")
        dataset = client.get_dataset(dataset_name)
        logger.info(
            f"Dataset '{dataset_name}' retrieved successfully with {len(dataset.items)} items"
        )
        return dataset

    except Exception as e:
        logger.error(f"Failed to retrieve dataset '{dataset_name}': {e}")
        raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found") from e


def filter_queries(
    queries: List[Any],
    filter_type: DatasetFilterType,
    filter_params: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Filter FinDER queries based on specified criteria.

    Args:
        queries: List of Query objects to filter
        filter_type: Type of filtering to apply
        filter_params: Parameters for filtering (required for some filter types)

    Returns:
        Filtered list of Query objects

    Raises:
        ValueError: If filter_params missing for filter types that require it
    """
    logger.info(f"Filtering {len(queries)} queries with filter_type={filter_type}")

    if filter_type == DatasetFilterType.ALL:
        return queries

    elif filter_type == DatasetFilterType.REASONING_REQUIRED:
        # Filter for queries that require multi-hop reasoning
        filtered = [
            q
            for q in queries
            if hasattr(q, "metadata")
            and hasattr(q.metadata, "reasoning_required")
            and q.metadata.reasoning_required
        ]
        logger.info(f"Filtered to {len(filtered)} reasoning-required queries")
        return filtered

    elif filter_type == DatasetFilterType.NO_REASONING:
        # Filter for queries without reasoning requirement
        filtered = [
            q
            for q in queries
            if hasattr(q, "metadata")
            and (
                not hasattr(q.metadata, "reasoning_required")
                or not q.metadata.reasoning_required
            )
        ]
        logger.info(f"Filtered to {len(filtered)} no-reasoning queries")
        return filtered

    elif filter_type == DatasetFilterType.BY_CATEGORY:
        if not filter_params or "category" not in filter_params:
            raise ValueError(
                "filter_params must include 'category' for BY_CATEGORY filter"
            )

        category = filter_params["category"]
        filtered = [
            q
            for q in queries
            if hasattr(q, "metadata")
            and hasattr(q.metadata, "financial_subdomain")
            and q.metadata.financial_subdomain == category
        ]
        logger.info(f"Filtered to {len(filtered)} queries in category '{category}'")
        return filtered

    elif filter_type == DatasetFilterType.BY_QUERY_TYPE:
        if not filter_params or "query_type" not in filter_params:
            raise ValueError(
                "filter_params must include 'query_type' for BY_QUERY_TYPE filter"
            )

        query_type = filter_params["query_type"]
        filtered = [
            q
            for q in queries
            if hasattr(q, "metadata")
            and hasattr(q.metadata, "query_type")
            and q.metadata.query_type == query_type
        ]
        logger.info(f"Filtered to {len(filtered)} queries of type '{query_type}'")
        return filtered

    elif filter_type == DatasetFilterType.BY_COMPLEXITY:
        if not filter_params:
            raise ValueError(
                "filter_params must include 'min_complexity' or 'max_complexity'"
            )

        min_complexity = filter_params.get("min_complexity")
        max_complexity = filter_params.get("max_complexity")

        filtered = queries
        if min_complexity is not None:
            filtered = [
                q
                for q in filtered
                if hasattr(q, "metadata")
                and hasattr(q.metadata, "complexity")
                and q.metadata.complexity >= min_complexity
            ]

        if max_complexity is not None:
            filtered = [
                q
                for q in filtered
                if hasattr(q, "metadata")
                and hasattr(q.metadata, "complexity")
                and q.metadata.complexity <= max_complexity
            ]

        logger.info(f"Filtered to {len(filtered)} queries by complexity")
        return filtered

    else:
        logger.warning(f"Unknown filter_type: {filter_type}, returning all queries")
        return queries


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _create_dataset_item_batch(
    client: Langfuse, dataset_name: str, items: List[Dict[str, Any]]
) -> int:
    """
    Create a batch of dataset items with retry logic.

    Args:
        client: Initialized Langfuse client
        dataset_name: Name of the dataset
        items: List of transformed dataset items (with input, expected_output, metadata)

    Returns:
        Number of successfully created items

    Raises:
        Exception: If batch creation fails after retries
    """
    successful_count = 0

    for item in items:
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item.get("metadata", {}),
            )
            successful_count += 1
        except Exception as e:
            logger.error(f"Failed to create dataset item: {e}")
            # Continue with other items even if one fails

    return successful_count


def upload_finder_dataset(
    client: Langfuse,
    config: LangfuseDatasetConfig,
    queries: List[Any],
    mapping: Optional[DatasetItemMapping] = None,
    batch_size: int = 50,
    rate_limit_delay: float = 0.5,
) -> Dict[str, Any]:
    """
    Upload FinDER dataset to Langfuse with batching and progress logging.

    This is the main orchestration function for dataset upload, handling:
    - Dataset creation
    - Query filtering
    - Query transformation
    - Batched upload with rate limiting
    - Progress reporting
    - Metadata preservation

    Args:
        client: Initialized Langfuse client
        config: Dataset configuration
        queries: List of FinDER Query objects
        mapping: Optional custom mapping (uses default if not provided)
        batch_size: Number of items per batch
        rate_limit_delay: Delay in seconds between batches

    Returns:
        Dict with upload summary:
            - dataset_name: str
            - total_items: int
            - successful_items: int
            - failed_items: int
            - upload_time_seconds: float
            - errors: List[str]

    Raises:
        DatasetUploadError: If upload fails critically
    """
    start_time = time.time()
    errors = []

    try:
        logger.info(f"Starting upload for dataset: {config.name}")

        # Create dataset
        create_dataset(client, config)

        # Filter queries based on config
        logger.info(
            f"Filtering {len(queries)} queries with filter_type={config.filter_type}"
        )
        filtered_queries = filter_queries(
            queries, config.filter_type, config.filter_params
        )

        # Apply max_items limit if specified
        if config.max_items and len(filtered_queries) > config.max_items:
            logger.info(
                f"Limiting to {config.max_items} items (from {len(filtered_queries)})"
            )
            filtered_queries = filtered_queries[: config.max_items]

        total_items = len(filtered_queries)
        logger.info(f"Uploading {total_items} items to dataset '{config.name}'")

        # Use default mapping if not provided
        if mapping is None:
            mapping = DatasetItemMapping()

        # Transform queries to Langfuse dataset items
        logger.info("Transforming queries to Langfuse dataset items")
        dataset_items = []
        for query in filtered_queries:
            try:
                transformed = mapping.transform_query(query)

                # Add version tag to metadata if specified
                if config.version_tag:
                    if "metadata" not in transformed:
                        transformed["metadata"] = {}
                    transformed["metadata"]["version"] = config.version_tag

                dataset_items.append(transformed)
            except Exception as e:
                error_msg = f"Failed to transform query {query.id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Upload in batches
        successful_items = 0
        num_batches = (len(dataset_items) + batch_size - 1) // batch_size

        for i in range(0, len(dataset_items), batch_size):
            batch_num = i // batch_size + 1
            batch = dataset_items[i : i + batch_size]

            logger.info(
                f"Uploading batch {batch_num}/{num_batches} ({len(batch)} items)"
            )

            try:
                batch_successful = _create_dataset_item_batch(
                    client, config.name, batch
                )
                successful_items += batch_successful

                # Progress reporting
                progress_pct = (i + len(batch)) / len(dataset_items) * 100
                logger.info(
                    f"Progress: {progress_pct:.1f}% ({successful_items}/{total_items} items)"
                )

            except Exception as e:
                error_msg = f"Batch {batch_num} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            # Rate limiting delay between batches
            if batch_num < num_batches:
                time.sleep(rate_limit_delay)

        # Final flush to ensure all events are sent
        logger.info("Flushing Langfuse client")
        client.flush()

        upload_time = time.time() - start_time
        failed_items = total_items - successful_items

        result = {
            "dataset_name": config.name,
            "total_items": total_items,
            "successful_items": successful_items,
            "failed_items": failed_items,
            "upload_time_seconds": upload_time,
            "errors": errors,
        }

        logger.info(
            f"Dataset upload complete: {successful_items}/{total_items} items in {upload_time:.2f}s"
        )

        return result

    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise DatasetUploadError(f"Failed to upload dataset: {e}") from e
