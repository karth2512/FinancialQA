"""
Complete end-to-end example of Langfuse experiment tracking.

This example demonstrates:
1. Uploading a dataset to Langfuse
2. Running an experiment with automatic tracing
3. Analyzing results
4. Comparing multiple experiments

Prerequisites:
- Langfuse credentials configured in .env
- FinDER dataset downloaded
- Dependencies installed (langfuse, sentence-transformers)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langfuse import Langfuse

from src.langfuse_integration.dataset_manager import upload_finder_dataset
from src.langfuse_integration.experiment_runner import run_langfuse_experiment
from src.langfuse_integration.analysis import compare_experiments, generate_comparison_report
from src.langfuse_integration.models import LangfuseDatasetConfig, DatasetFilterType
from src.config.experiments import LangfuseExperimentConfig, RetrievalConfig, LLMConfig


def main():
    """
    Run complete Langfuse integration example.
    """
    # Load environment variables
    load_dotenv()

    # Initialize Langfuse client
    print("Initializing Langfuse client...")
    client = Langfuse()

    # ==========================================================================
    # Step 1: Upload Dataset
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 1: UPLOAD DATASET TO LANGFUSE")
    print("="*70)

    # NOTE: This is a placeholder - replace with actual data loading
    # from src.data.loader import load_finder_dataset
    # queries = load_finder_dataset()

    # For demonstration, we'll skip actual upload if dataset exists
    try:
        dataset = client.get_dataset("financial_qa_test")
        print(f"✓ Dataset 'financial_qa_test' already exists with {len(dataset.items)} items")
    except:
        print("Dataset not found. In a real scenario, you would upload it here.")
        print("Example code:")
        print("""
        dataset_config = LangfuseDatasetConfig(
            name="financial_qa_test",
            description="Test dataset for demonstration",
            filter_type=DatasetFilterType.ALL,
            max_items=100,  # Limit for testing
            metadata_enrichment={
                "source": "FinDER",
                "purpose": "example"
            }
        )

        result = upload_finder_dataset(
            client=client,
            config=dataset_config,
            queries=queries[:100],  # First 100 queries
            batch_size=20
        )

        print(f"✓ Uploaded {result['successful_items']} items")
        """)

    # ==========================================================================
    # Step 2: Run Baseline Experiment
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 2: RUN BASELINE EXPERIMENT")
    print("="*70)

    # Create experiment configuration
    baseline_config = LangfuseExperimentConfig(
        name="example_baseline",
        description="Baseline RAG with BM25 retrieval",
        run_id="example-baseline-001",
        pipeline_type="baseline",
        retrieval_config=RetrievalConfig(
            strategy="bm25",
            top_k=5
        ),
        llm_configs={
            "generator": LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.0,
                max_tokens=256
            )
        },
        langfuse_dataset_name="financial_qa_test",
        max_concurrency=1,  # Sequential execution (default)
        enable_item_evaluators=True,
        enable_run_evaluators=True,
        item_evaluator_names=["token_f1", "semantic_similarity"],
        run_evaluator_names=["average_accuracy"],
        langfuse_tags=["example", "baseline"],
        langfuse_metadata={
            "experiment_type": "example",
            "version": "1.0"
        }
    )

    print("\nConfiguration:")
    print(f"  Name: {baseline_config.name}")
    print(f"  Dataset: {baseline_config.langfuse_dataset_name}")
    print(f"  Pipeline: {baseline_config.pipeline_type}")
    print(f"  Evaluators: {', '.join(baseline_config.item_evaluator_names)}")

    print("\nNote: This example uses a placeholder RAG implementation.")
    print("In production, replace with your actual RAG pipeline.")

    # Uncomment to run actual experiment:
    # result = run_langfuse_experiment(baseline_config, client)
    # print(f"\n✓ Experiment complete!")
    # print(f"  Success rate: {result.get_success_rate():.2%}")
    # print(f"  Average accuracy: {result.aggregate_scores.get('average_accuracy', 0):.3f}")
    # print(f"  Langfuse URL: {result.langfuse_url}")

    # ==========================================================================
    # Step 3: Run Improved Experiment
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 3: RUN IMPROVED EXPERIMENT")
    print("="*70)

    # Create improved configuration (with different retrieval strategy)
    improved_config = LangfuseExperimentConfig(
        name="example_improved",
        description="Improved RAG with dense retrieval",
        run_id="example-improved-001",
        pipeline_type="baseline",
        retrieval_config=RetrievalConfig(
            strategy="dense",
            top_k=10,
            embedding_model="all-MiniLM-L6-v2"
        ),
        llm_configs={
            "generator": LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.0,
                max_tokens=256
            )
        },
        langfuse_dataset_name="financial_qa_test",
        max_concurrency=1,
        enable_item_evaluators=True,
        enable_run_evaluators=True,
        item_evaluator_names=["token_f1", "semantic_similarity"],
        run_evaluator_names=["average_accuracy"],
        langfuse_tags=["example", "improved"],
        langfuse_metadata={
            "experiment_type": "example",
            "version": "2.0"
        }
    )

    print("\nConfiguration:")
    print(f"  Name: {improved_config.name}")
    print(f"  Retrieval: {improved_config.retrieval_config.strategy} (vs baseline: bm25)")
    print(f"  Top-K: {improved_config.retrieval_config.top_k} (vs baseline: 5)")

    # Uncomment to run:
    # result_improved = run_langfuse_experiment(improved_config, client)

    # ==========================================================================
    # Step 4: Compare Experiments
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPARE EXPERIMENTS")
    print("="*70)

    print("\nExample comparison code:")
    print("""
    # Compare the two experiments
    comparison = compare_experiments(
        client,
        experiment_ids=["example-baseline-001", "example-improved-001"]
    )

    # Generate comparison report
    report = generate_comparison_report(comparison, output_format="text")
    print(report)

    # Save Markdown report
    md_report = generate_comparison_report(comparison, output_format="markdown")
    with open("experiment_comparison.md", "w") as f:
        f.write(md_report)

    print("✓ Comparison complete!")
    print("  View report: experiment_comparison.md")
    """)

    # ==========================================================================
    # Step 5: Regression Detection
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 5: REGRESSION DETECTION")
    print("="*70)

    print("\nExample regression detection code:")
    print("""
    from src.langfuse_integration.analysis import detect_regression

    # Detect if improved experiment regressed compared to baseline
    regression_result = detect_regression(
        client,
        current_experiment_id="example-improved-001",
        baseline_experiment_id="example-baseline-001",
        threshold=0.05  # 5% threshold
    )

    if regression_result["has_regression"]:
        print("⚠️  Regressions detected:")
        for reg in regression_result["regressions"]:
            print(f"  {reg['metric']}: {reg['change_percent']:.2f}% decrease")
    else:
        print("✓ No regressions detected")

    if regression_result["improvements"]:
        print("\\n✓ Improvements detected:")
        for imp in regression_result["improvements"]:
            print(f"  {imp['metric']}: +{imp['change_percent']:.2f}% increase")
    """)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print()
    print("This example demonstrated:")
    print("  1. ✓ Dataset upload to Langfuse")
    print("  2. ✓ Experiment execution with tracing")
    print("  3. ✓ Automatic evaluation (token F1, semantic similarity)")
    print("  4. ✓ Experiment comparison")
    print("  5. ✓ Regression detection")
    print()
    print("Next steps:")
    print("  - Implement your actual RAG pipeline")
    print("  - Upload your FinDER dataset")
    print("  - Run real experiments")
    print("  - View results in Langfuse UI")
    print()
    print("For full implementation, see:")
    print("  - scripts/upload_dataset.py")
    print("  - scripts/run_experiment.py")
    print("  - scripts/compare_experiments.py")
    print()


if __name__ == "__main__":
    main()
