"""
Experiment execution orchestration with Langfuse SDK integration.

This module provides the main interface for running RAG experiments with
automatic tracing, evaluation, and result aggregation.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from langfuse import Langfuse

from .models import ExperimentRunOutput, ItemExecutionResult, EvaluationScore
from .config import validate_langfuse_credentials, load_langfuse_config
from .exceptions import ExperimentExecutionError, DatasetNotFoundError
from ..config.experiments import LangfuseExperimentConfig

logger = logging.getLogger(__name__)


def create_task_function(config: LangfuseExperimentConfig) -> Callable:
    """
    Create a task function based on pipeline type.

    The task function receives a dataset item and returns the RAG pipeline output.
    This factory pattern allows different pipeline types (baseline, query_expansion)
    to be created based on config.

    Args:
        config: Experiment configuration

    Returns:
        Callable task function with signature: (*, item, **kwargs) -> Dict[str, Any]

    Raises:
        ValueError: If pipeline_type is not supported
    """
    pipeline_type = config.pipeline_type

    if pipeline_type == "baseline":
        return _create_baseline_task(config)
    elif pipeline_type == "query_expansion":
        return _create_query_expansion_task(config)
    else:
        raise ValueError(
            f"Unsupported pipeline_type: {pipeline_type}. "
            f"Valid types: baseline, query_expansion"
        )


def _create_baseline_task(config: LangfuseExperimentConfig) -> Callable:
    """
    Create task function for baseline RAG pipeline.

    Args:
        config: Experiment configuration

    Returns:
        Task function for baseline RAG
    """
    # Import RAG components
    from ..rag.baseline import BaselineRAG
    from ..retrieval.bm25 import BM25Retriever
    from ..retrieval.dense import DenseRetriever
    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.graphrag import GraphRAGRetriever
    from ..data_handler.loader import FinDERLoader
    from ..data_handler.models import Query
    from ..utils.llm_client import create_llm_client

    # Initialize retriever based on config
    retrieval_config = config.retrieval_config
    strategy = retrieval_config.strategy.lower()

    # Load corpus for retrieval
    logger.info(f"Loading corpus for {strategy} retrieval...")
    loader = FinDERLoader()
    corpus = loader.load_corpus()
    logger.info(f"Loaded {len(corpus)} passages for retrieval")

    # Create retriever based on strategy
    retriever_config = {
        "k1": retrieval_config.k1,
        "b": retrieval_config.b,
        "top_k": retrieval_config.top_k,
    }

    if strategy == "bm25":
        retriever = BM25Retriever(retriever_config)
    elif strategy == "dense":
        retriever = DenseRetriever(retriever_config)
    elif strategy == "hybrid":
        retriever = HybridRetriever(retriever_config)
    elif strategy == "graphrag":
        retriever_config = {
            "graphrag_root": retrieval_config.graphrag_root or "./data/graphrag",
            "method": retrieval_config.graphrag_method or "local",
            "top_k": retrieval_config.top_k,
            "community_level": retrieval_config.graphrag_community_level or 2,
            "response_type": retrieval_config.graphrag_response_type or "Multiple Paragraphs",
        }
        retriever = GraphRAGRetriever(retriever_config)
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    # Index corpus
    logger.info(f"Indexing corpus with {strategy} retriever...")
    retriever.index_corpus(corpus)

    # Initialize LLM client
    llm_config = config.llm_configs.get("generator")
    if llm_config is None:
        raise ValueError("Generator LLM config not found in config.llm_configs")

    llm_client = create_llm_client(llm_config.model_dump())

    # Create RAG pipeline
    rag_pipeline = BaselineRAG(retriever=retriever, llm_client=llm_client)
    logger.info(f"Baseline RAG pipeline initialized with {strategy} retrieval")

    def baseline_task(*, item, **kwargs):
        """
        Execute baseline RAG pipeline for a single query.

        Args:
            item: Dataset item with input field containing query
                  When using dataset.run_experiment(), item is DatasetItemClient
                  When using client.run_experiment() with local data, item is dict
            **kwargs: Additional context from Langfuse

        Returns:
            Dict with answer, retrieved_passages, and metadata
        """
        # Extract query text from input
        if isinstance(item, dict):
            # Local data format: {"input": {...}, "expected_output": ...}
            input_data = item.get("input", {})
        else:
            # DatasetItemClient format: item.input, item.expected_output
            input_data = item.input

        query_text = input_data.get("text") or input_data.get("question")

        # Create Query object
        query = Query(
            id=input_data.get("id", "unknown"),
            text=query_text,
            expected_answer="a",   # Not used but needs to have value for post init checks 
            expected_evidence=["a"],  # Not used but needs to have value for post init checks 
            metadata=None,
        )

        # Run RAG pipeline
        rag_result = rag_pipeline.process_query(
            query=query,
            top_k=retrieval_config.top_k
        )

        # Format retrieved passages for Langfuse
        retrieved_passages = [
            {
                "text": rp.passage.text,
                "score": rp.score,
                "rank": rp.rank,
                "passage_id": rp.passage.id,
            }
            for rp in rag_result["retrieval_result"].retrieved_passages
        ]

        result = {
            "answer": rag_result["generated_answer"],
            "retrieved_passages": retrieved_passages,
            "retrieval_strategy": strategy,
            "model": llm_config.model,
            "latency_seconds": rag_result["latency_seconds"],
            "cost_estimate": rag_result["cost_estimate"],
            "prompt_tokens": rag_result["prompt_tokens"],
            "completion_tokens": rag_result["completion_tokens"],
        }

        return result

    return baseline_task


def _create_query_expansion_task(config: LangfuseExperimentConfig) -> Callable:
    """
    Create task function for query expansion RAG pipeline.

    This function:
    1. Loads corpus and builds retrieval index (once)
    2. Creates expander and generator LLM clients
    3. Instantiates QueryExpansionRAG pipeline
    4. Returns closure that processes each dataset item

    Args:
        config: Experiment configuration

    Returns:
        Task function for query expansion RAG
    """
    # Import RAG components
    from ..rag.query_expansion import QueryExpansionRAG
    from ..retrieval.bm25 import BM25Retriever
    from ..retrieval.dense import DenseRetriever
    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.graphrag import GraphRAGRetriever
    from ..data_handler.loader import FinDERLoader
    from ..data_handler.models import Query
    from ..utils.llm_client import create_llm_client

    # Load corpus (once for all queries)
    logger.info("Loading FinDER corpus...")
    loader = FinDERLoader()
    corpus = loader.load_corpus()
    logger.info(f"Loaded {len(corpus)} passages")

    # Create retriever based on strategy
    retrieval_config = config.retrieval_config
    strategy = retrieval_config.strategy.lower()

    retriever_config = {
        "k1": retrieval_config.k1,
        "b": retrieval_config.b,
        "top_k": retrieval_config.top_k,
    }

    if strategy == "bm25":
        retriever = BM25Retriever(retriever_config)
    elif strategy == "dense":
        retriever = DenseRetriever(retriever_config)
    elif strategy == "hybrid":
        retriever = HybridRetriever(retriever_config)
    elif strategy == "graphrag":
        retriever_config = {
            "graphrag_root": retrieval_config.graphrag_root or "./data/graphrag",
            "method": retrieval_config.graphrag_method or "local",
            "top_k": retrieval_config.top_k,
            "community_level": retrieval_config.graphrag_community_level or 2,
            "response_type": retrieval_config.graphrag_response_type or "Multiple Paragraphs",
        }
        retriever = GraphRAGRetriever(retriever_config)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # Index corpus (once)
    logger.info(f"Building {strategy} index...")
    retriever.index_corpus(corpus)
    logger.info("Index built successfully")

    # Create LLM clients
    expander_config = config.llm_configs.get("expander")
    generator_config = config.llm_configs.get("generator")

    if expander_config is None:
        raise ValueError("Expander LLM config not found in config.llm_configs")
    if generator_config is None:
        raise ValueError("Generator LLM config not found in config.llm_configs")

    expander_llm = create_llm_client(expander_config.model_dump())
    generator_llm = create_llm_client(generator_config.model_dump())

    logger.info(f"Expander LLM: {expander_config.provider}/{expander_config.model}")
    logger.info(f"Generator LLM: {generator_config.provider}/{generator_config.model}")

    # Get num_expanded_queries from hyperparameters
    num_expanded_queries = config.hyperparameters.get("num_expanded_queries", 3)
    logger.info(f"Query expansion: {num_expanded_queries} additional queries per input")

    # Create RAG pipeline
    rag_pipeline = QueryExpansionRAG(
        retriever=retriever,
        expander_llm=expander_llm,
        generator_llm=generator_llm,
        num_expanded_queries=num_expanded_queries,
    )

    def query_expansion_task(*, item, **kwargs):
        """
        Execute query expansion RAG pipeline for a single query.

        Args:
            item: Dataset item with input field containing query
            **kwargs: Additional context from Langfuse

        Returns:
            Dict with answer, retrieved_passages, expansion_metadata, and metadata
        """
        # Extract query text from input
        if isinstance(item, dict):
            input_data = item.get("input", {})
        else:
            input_data = item.input

        query_text = input_data.get("text") or input_data.get("question")

        # Create Query object
        query = Query(
            id=input_data.get("id", "unknown"),
            text=query_text,
            expected_answer="a", # Not used but needs to have value for post init checks 
            expected_evidence=["a"],  # Not used but needs to have value for post init checks 
            metadata=None,
        )

        # Process query through RAG pipeline
        rag_result = rag_pipeline.process_query(query, top_k=retrieval_config.top_k)

        # Format retrieved passages for Langfuse
        retrieved_passages = [
            {
                "passage_id": p.passage.id,
                "text": p.passage.text,
                "score": p.score,
                "rank": p.rank,
            }
            for p in rag_result["retrieval_result"].retrieved_passages
        ]

        result = {
            "answer": rag_result["generated_answer"],
            "retrieved_passages": retrieved_passages,
            "latency_seconds": rag_result["latency_seconds"],
            "cost_estimate": rag_result.get("cost_estimate", 0.0),
            "expansion_metadata": rag_result["expansion_metadata"],
            "metadata": {
                "pipeline_type": "query_expansion",
                "retrieval_strategy": strategy,
                "expander_model": expander_config.model,
                "generator_model": generator_config.model,
                "num_expanded_queries": num_expanded_queries,
            },
        }

        return result

    return query_expansion_task


# Multi-agent task creation removed - not implemented
# If multi-agent RAG is needed in the future, implement fresh based on requirements


def _initialize_langfuse_client(
    config: LangfuseExperimentConfig, client: Optional[Langfuse] = None
) -> Langfuse:
    """
    Initialize Langfuse client with credential validation.

    Args:
        config: Experiment configuration
        client: Optional existing client

    Returns:
        Initialized Langfuse client

    Raises:
        CredentialError: If credentials are invalid
    """
    if client is not None:
        logger.info("Using provided Langfuse client")
        return client

    logger.info("Initializing new Langfuse client")

    # Validate credentials
    validate_langfuse_credentials()

    # Create client with config settings
    client = Langfuse(
        enabled=config.enable_tracing,
        flush_at=config.flush_batch_size,
        flush_interval=config.flush_interval,
    )

    logger.info("Langfuse client initialized successfully")
    return client


def _load_experiment_data(config: LangfuseExperimentConfig, client: Langfuse) -> Any:
    """
    Load experiment data from Langfuse dataset or local source.

    Args:
        config: Experiment configuration
        client: Langfuse client

    Returns:
        DatasetClient object if using Langfuse dataset,
        or list of dict items if using local data

    Raises:
        DatasetNotFoundError: If Langfuse dataset not found
        FileNotFoundError: If local data file not found
    """
    if config.langfuse_dataset_name:
        logger.info(f"Loading Langfuse dataset: {config.langfuse_dataset_name}")
        try:
            dataset = client.get_dataset(config.langfuse_dataset_name)
            logger.info(f"Loaded dataset with {len(dataset.items)} items")
            return dataset  # Return DatasetClient object directly
        except Exception as e:
            raise DatasetNotFoundError(config.langfuse_dataset_name) from e

    elif config.use_local_data:
        logger.info("Loading local data")
        # TODO: Implement local data loading
        # from src.data_handler.loader import load_finder_queries
        # queries = load_finder_queries(config.local_data_path)
        # return [format_query_to_item(q) for q in queries]

        # Placeholder: Return empty list
        logger.warning("Local data loading not implemented - using empty dataset")
        return []

    else:
        raise ValueError("Either langfuse_dataset_name or use_local_data must be set")


def _transform_experiment_result(
    sdk_result: Any,
    config: LangfuseExperimentConfig,
    start_time: datetime,
    end_time: datetime,
) -> ExperimentRunOutput:
    """
    Transform Langfuse SDK experiment result to ExperimentRunOutput.

    Args:
        sdk_result: Result from langfuse.run_experiment()
        config: Experiment configuration
        start_time: Experiment start timestamp
        end_time: Experiment end timestamp

    Returns:
        ExperimentRunOutput with all results and metadata
    """
    logger.info("Transforming SDK result to ExperimentRunOutput")

    # Extract item results
    item_results = []
    successful_items = 0
    failed_items = 0

    for item_result in sdk_result.item_results:
        try:
            # Convert SDK item result to ItemExecutionResult
            item_exec_result = ItemExecutionResult(
                item_id=getattr(item_result, "item_id", "unknown"),
                input=getattr(item_result, "input", {}),
                output=getattr(item_result, "output", {}),
                expected_output=getattr(item_result, "expected_output", None),
                evaluations=[
                    EvaluationScore(
                        name=eval.name,
                        value=eval.value,
                        data_type=(
                            eval.data_type if hasattr(eval, "data_type") else "NUMERIC"
                        ),
                        comment=eval.comment if hasattr(eval, "comment") else None,
                    )
                    for eval in getattr(item_result, "evaluations", [])
                ],
                trace_id=getattr(item_result, "trace_id", "unknown"),
                latency_seconds=getattr(item_result, "latency", 0.0),
                error=getattr(item_result, "error", None),
            )

            item_results.append(item_exec_result)

            if item_exec_result.error is None:
                successful_items += 1
            else:
                failed_items += 1

        except Exception as e:
            logger.error(f"Failed to transform item result: {e}")
            failed_items += 1

    # Extract aggregate scores from run evaluators
    aggregate_scores = {}
    if hasattr(sdk_result, "aggregate_scores"):
        aggregate_scores = sdk_result.aggregate_scores
    elif hasattr(sdk_result, "run_evaluations"):
        for eval in sdk_result.run_evaluations:
            aggregate_scores[eval.name] = eval.value

    # Calculate total latency
    total_latency = (end_time - start_time).total_seconds()

    # Determine status
    if failed_items == 0:
        status = "completed"
    elif successful_items > 0:
        status = "partial"
    else:
        status = "failed"

    # Build Langfuse URL
    langfuse_host = load_langfuse_config().get("host", "https://cloud.langfuse.com")
    langfuse_url = (
        f"{langfuse_host}/experiments/{getattr(sdk_result, 'run_id', config.run_id)}"
    )

    result = ExperimentRunOutput(
        run_id=getattr(sdk_result, "run_id", config.run_id),
        experiment_name=config.name,
        dataset_name=config.langfuse_dataset_name,
        dataset_run_id=getattr(sdk_result, "dataset_run_id", None),
        config=config.model_dump(),
        item_results=item_results,
        aggregate_scores=aggregate_scores,
        total_items=len(item_results),
        successful_items=successful_items,
        failed_items=failed_items,
        start_time=start_time,
        end_time=end_time,
        total_latency_seconds=total_latency,
        status=status,
        langfuse_url=langfuse_url,
    )

    logger.info(
        f"Experiment result: {successful_items}/{len(item_results)} successful, status={status}"
    )
    return result


def run_langfuse_experiment(
    config: LangfuseExperimentConfig, langfuse_client: Optional[Langfuse] = None
) -> ExperimentRunOutput:
    """
    Execute a complete RAG experiment with automatic Langfuse tracing.

    This is the primary entry point for running experiments. It orchestrates:
    - Dataset loading (from Langfuse or local data)
    - Task function creation from RAG pipeline
    - Evaluator registration
    - Experiment execution via langfuse.run_experiment()
    - Result aggregation and persistence

    Args:
        config: Complete experiment configuration with Langfuse settings
        langfuse_client: Optional Langfuse client (creates new if not provided)

    Returns:
        ExperimentRunOutput with complete results, traces, and metrics

    Raises:
        ValueError: If config is invalid or required dependencies missing
        CredentialError: If Langfuse credentials are invalid
        DatasetNotFoundError: If dataset not found
        ExperimentExecutionError: If experiment execution fails critically
    """
    start_time = datetime.now()

    try:
        logger.info(f"Starting experiment: {config.name}")
        logger.info(f"Run ID: {config.run_id}")
        logger.info(f"Pipeline type: {config.pipeline_type}")
        logger.info(f"Max concurrency: {config.max_concurrency}")

        # Initialize Langfuse client
        client = _initialize_langfuse_client(config, langfuse_client)

        # Load experiment data (DatasetClient for Langfuse datasets, list for local data)
        data = _load_experiment_data(config, client)

        # Create task function
        logger.info("Creating task function from RAG pipeline")
        task = create_task_function(config)

        # Register evaluators
        logger.info("Registering evaluators")
        from .evaluators import register_evaluators

        item_evaluators, run_evaluators = register_evaluators(config)

        # Build metadata and tags for propagation
        metadata = config.langfuse_metadata.copy()
        metadata.update(
            {
                "experiment_name": config.name,
                "run_id": config.run_id,
                "pipeline_type": config.pipeline_type,
                "retrieval_strategy": config.retrieval_config.strategy,
            }
        )

        tags = config.langfuse_tags.copy()
        tags.extend(
            [
                f"experiment:{config.name}",
                f"pipeline:{config.pipeline_type}",
                f"retrieval:{config.retrieval_config.strategy}",
            ]
        )

        # Run experiment via Langfuse SDK
        logger.info("Executing experiment via Langfuse SDK")
        logger.info(f"Item evaluators: {len(item_evaluators)}")
        logger.info(f"Run evaluators: {len(run_evaluators)}")

        # Check if data is a DatasetClient or list
        if hasattr(data, 'run_experiment'):
            # Using Langfuse dataset - call dataset.run_experiment()
            logger.info(f"Running experiment via dataset.run_experiment() on {config.langfuse_dataset_name}")
            sdk_result = data.run_experiment(
                name=config.name,
                run_name=config.run_id,
                description=config.description,
                task=task,
                evaluators=item_evaluators if config.enable_item_evaluators else [],
                run_evaluators=run_evaluators if config.enable_run_evaluators else [],
                max_concurrency=config.max_concurrency,
                metadata=metadata,
            )
        else:
            # Using local data list - call client.run_experiment()
            logger.info("Running experiment via client.run_experiment() with local data")
            sdk_result = client.run_experiment(
                name=config.name,
                run_name=config.run_id,
                description=config.description,
                data=data,
                task=task,
                evaluators=item_evaluators if config.enable_item_evaluators else [],
                run_evaluators=run_evaluators if config.enable_run_evaluators else [],
                max_concurrency=config.max_concurrency,
                metadata=metadata,
            )

        # Flush to ensure all traces are sent
        logger.info("Flushing Langfuse client")
        client.flush()

        end_time = datetime.now()

        # Transform SDK result to ExperimentRunOutput
        result = _transform_experiment_result(sdk_result, config, start_time, end_time)

        logger.info(f"Experiment completed: {result.experiment_name}")
        logger.info(f"Success rate: {result.get_success_rate():.2%}")
        logger.info(f"Total time: {result.total_latency_seconds:.2f}s")

        return result

    except Exception as e:
        logger.error(f"Experiment execution failed: {e}", exc_info=True)
        end_time = datetime.now()

        # Try to flush any partial data
        if langfuse_client:
            try:
                langfuse_client.flush()
            except:
                pass

        raise ExperimentExecutionError(
            message=f"Experiment '{config.name}' failed: {e}",
            experiment_name=config.name,
            run_id=config.run_id,
        ) from e
