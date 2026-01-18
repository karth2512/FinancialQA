"""
Query Expansion RAG pipeline.

This module implements a query expansion approach:
query → expand (M variants) → retrieve (M+1 times) → pool & deduplicate → generate.
"""

import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from src.data_handler.models import Query, RetrievalResult, RetrievedPassage
from src.retrieval.base import RetrieverBase
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.llm_client import LLMClient, create_llm_client


class ExpandedQueries(BaseModel):
    """Pydantic model for structured LLM output of query expansions."""

    queries: List[str] = Field(
        description="List of alternative phrasings of the original question",
        min_length=1,
        max_length=10,
    )


class QueryExpansionRAG:
    """RAG pipeline with LLM-based query expansion before retrieval."""

    def __init__(
        self,
        retriever: RetrieverBase,
        expander_llm: LLMClient,
        generator_llm: LLMClient,
        num_expanded_queries: int = 3,
    ):
        """
        Initialize query expansion RAG pipeline.

        Args:
            retriever: Retrieval strategy (BM25, Dense, or Hybrid)
            expander_llm: LLM client for query expansion
            generator_llm: LLM client for answer generation
            num_expanded_queries: Number of query variants to generate (default: 3)
        """
        self.retriever = retriever
        self.expander_llm = expander_llm
        self.generator_llm = generator_llm
        self.num_expanded_queries = num_expanded_queries

    def process_query(
        self,
        query: Query,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process a single query through the query expansion RAG pipeline.

        Steps:
        1. Expand query into M variants using structured output
        2. Retrieve for each query (original + M expansions)
        3. Pool and deduplicate passages by ID
        4. Re-rank by score
        5. Generate answer with pooled passages

        Args:
            query: Query object to process
            top_k: Number of passages to retrieve per query variant

        Returns:
            Dict with generated_answer, retrieval_result, latency, cost, expansion_metadata
        """
        start_time = time.time()

        # Step 1: Expand query
        expansion_start = time.time()
        expanded_queries = self._expand_query(query.text)
        expansion_latency = time.time() - expansion_start

        # Count expansion tokens
        expansion_prompt_tokens = self.expander_llm.count_tokens(
            f"Generate {self.num_expanded_queries} alternative phrasings. Original: {query.text}"
        )
        expansion_completion_tokens = sum(
            self.expander_llm.count_tokens(q) for q in expanded_queries[1:]
        )
        expansion_cost = self.expander_llm.estimate_cost(
            expansion_prompt_tokens, expansion_completion_tokens
        )

        # Step 2: Retrieve for each query variant
        retrieval_start = time.time()
        all_retrieval_results = []
        for query_variant in expanded_queries:
            result = self.retriever.retrieve(query_text=query_variant, top_k=top_k)
            all_retrieval_results.append(result)
        retrieval_latency = time.time() - retrieval_start

        # Step 3: Pool and deduplicate
        pooled_result = self._pool_and_rerank(all_retrieval_results, query.id)

        # Step 4: Generate answer
        generation_start = time.time()
        prompt = self._construct_generation_prompt(query, pooled_result)
        prompt_tokens = self.generator_llm.count_tokens(prompt)

        try:
            generated_answer = self.generator_llm.generate(prompt)
            completion_tokens = self.generator_llm.count_tokens(generated_answer)
        except Exception as e:
            generated_answer = f"Error generating answer: {e}"
            completion_tokens = 0

        generation_latency = time.time() - generation_start

        # Step 5: Calculate costs
        generation_cost = self.generator_llm.estimate_cost(
            prompt_tokens, completion_tokens
        )
        total_cost = expansion_cost + generation_cost

        total_latency = time.time() - start_time

        return {
            "generated_answer": generated_answer,
            "retrieval_result": pooled_result,
            "latency_seconds": total_latency,
            "cost_estimate": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "expansion_metadata": {
                "original_query": query.text,
                "expanded_queries": expanded_queries[1:],  # Exclude original
                "num_expansions": len(expanded_queries) - 1,
                "expansion_latency_seconds": expansion_latency,
                "retrieval_latency_seconds": retrieval_latency,
                "generation_latency_seconds": generation_latency,
                "total_passages_before_dedup": sum(
                    len(r.retrieved_passages) for r in all_retrieval_results
                ),
                "total_passages_after_dedup": len(pooled_result.retrieved_passages),
                "expansion_cost": expansion_cost,
                "generation_cost": generation_cost,
            },
        }

    def _expand_query(self, query_text: str) -> List[str]:
        """
        Generate M query variants using expander LLM with structured output.

        Args:
            query_text: Original query text

        Returns:
            List of queries: [original_query, expanded_1, expanded_2, ...]
        """
        try:
            # Use structured output if LLM supports it
            if hasattr(self.expander_llm, "generate_structured"):
                # OpenAI/Anthropic with JSON schema
                prompt = f"""Generate {self.num_expanded_queries} alternative phrasings of this financial question.

Original Question: {query_text}

Requirements:
- Preserve the original meaning and intent
- Use different wording and terminology
- Cover different aspects or perspectives
- Maintain the same level of specificity"""

                result = self.expander_llm.generate_structured(
                    prompt=prompt, response_model=ExpandedQueries
                )
                expanded = result.queries[: self.num_expanded_queries]
            else:
                # Fallback to JSON mode for older LLM clients
                schema = ExpandedQueries.model_json_schema()
                prompt = f"""Generate {self.num_expanded_queries} alternative phrasings of this financial question.

Original Question: {query_text}

Return ONLY a JSON object matching this schema:
{schema}

Example:
{{"queries": ["What is the stock price?", "How much does the stock cost?", "What's the current share value?"]}}"""

                response = self.expander_llm.generate(prompt)
                result = ExpandedQueries.model_validate_json(response)
                expanded = result.queries[: self.num_expanded_queries]

            # Pad or truncate to exactly M queries
            if len(expanded) < self.num_expanded_queries:
                # Pad with original query
                expanded.extend([query_text] * (self.num_expanded_queries - len(expanded)))
            elif len(expanded) > self.num_expanded_queries:
                # Truncate
                expanded = expanded[: self.num_expanded_queries]

            # Return original + expansions
            return [query_text] + expanded

        except Exception as e:
            # Fallback: return original query repeated
            print(f"Query expansion failed: {e}. Using original query repeated.")
            return [query_text] * (self.num_expanded_queries + 1)

    def _pool_and_rerank(
        self,
        retrieval_results: List[RetrievalResult],
        query_id: str,
    ) -> RetrievalResult:
        """
        Deduplicate passages by ID, keep highest score, re-rank.

        Args:
            retrieval_results: List of retrieval results from multiple queries
            query_id: Query ID for the result

        Returns:
            RetrievalResult with pooled and re-ranked passages
        """
        # Create dict: passage_id -> RetrievedPassage
        passage_dict = {}

        for result in retrieval_results:
            for retrieved_passage in result.retrieved_passages:
                passage_id = retrieved_passage.passage.id

                if passage_id not in passage_dict:
                    passage_dict[passage_id] = retrieved_passage
                else:
                    # Keep passage with highest score
                    if retrieved_passage.score > passage_dict[passage_id].score:
                        passage_dict[passage_id] = retrieved_passage

        # Convert to list and sort by score (descending)
        pooled_passages = sorted(
            passage_dict.values(), key=lambda p: p.score, reverse=True
        )

        # Re-assign ranks
        for i, passage in enumerate(pooled_passages):
            passage.rank = i + 1

        # Create new RetrievalResult
        return RetrievalResult(
            query_id=query_id,
            retrieved_passages=pooled_passages,
            strategy="bm25",
            retrieval_time_seconds=sum(
                r.retrieval_time_seconds for r in retrieval_results
            ),
            metadata={
                "num_query_variants": len(retrieval_results),
                "original_strategy": (
                    retrieval_results[0].strategy if retrieval_results else "unknown"
                ),
            },
        )

    def _construct_generation_prompt(
        self,
        query: Query,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Construct prompt for answer generation (same as baseline).

        Args:
            query: Query object
            retrieval_result: Retrieved passages (pooled and deduplicated)

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a financial question-answering assistant. Answer the following question based on the provided evidence passages.",
            "",
            "Question: " + query.text,
            "",
            "Evidence Passages:",
        ]

        # Add retrieved passages (limit to top 10 to avoid token limits)
        for i, retrieved_passage in enumerate(
            retrieval_result.retrieved_passages[:10], start=1
        ):
            prompt_parts.append(
                f"\n[Passage {i}] (Score: {retrieved_passage.score:.3f})"
            )
            prompt_parts.append(retrieved_passage.passage.text)

        prompt_parts.extend(
            [
                "",
                "Instructions:",
                "- Answer the question based ONLY on the evidence passages provided above",
                "- Be concise and factual",
                "- If the evidence is insufficient to answer the question, say 'Insufficient evidence'",
                "- Do not make up information not present in the evidence",
                "",
                "Answer:",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_expansion_schema(self) -> Dict[str, Any]:
        """Get JSON schema for ExpandedQueries model."""
        return ExpandedQueries.model_json_schema()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QueryExpansionRAG":
        """
        Create query expansion RAG pipeline from configuration.

        Args:
            config: Configuration dict with:
                - retrieval_config: Retrieval configuration
                - llm_configs: Dict with 'expander' and 'generator' LLM configs
                - hyperparameters: Dict with 'num_expanded_queries' (optional)

        Returns:
            Initialized QueryExpansionRAG instance
        """
        # Create retriever based on strategy
        retrieval_config = config["retrieval_config"]
        strategy = retrieval_config.get("strategy", "bm25")

        if strategy == "bm25":
            retriever = BM25Retriever(retrieval_config)
        elif strategy == "dense":
            retriever = DenseRetriever(retrieval_config)
        elif strategy == "hybrid":
            retriever = HybridRetriever(retrieval_config)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        # Create LLM clients
        llm_configs = config.get("llm_configs", {})
        expander_llm = create_llm_client(llm_configs["expander"])
        generator_llm = create_llm_client(llm_configs["generator"])

        # Get num_expanded_queries from hyperparameters
        hyperparameters = config.get("hyperparameters", {})
        num_expanded_queries = hyperparameters.get("num_expanded_queries", 3)

        return cls(
            retriever=retriever,
            expander_llm=expander_llm,
            generator_llm=generator_llm,
            num_expanded_queries=num_expanded_queries,
        )
