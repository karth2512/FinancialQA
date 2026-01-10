#!/usr/bin/env python3
"""
Validation script to check that the baseline RAG system is properly set up.

Run this script after installing dependencies to verify the setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure stdout to use UTF-8 encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_imports():
    """Check that all required packages can be imported."""
    print("Checking package imports...")

    required_packages = [
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("rank_bm25", "Rank BM25"),
        ("chromadb", "ChromaDB"),
        ("langfuse", "Langfuse"),
        ("openai", "OpenAI"),
        ("pydantic", "Pydantic"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]

    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✓ All required packages found")
    return True


def check_environment():
    """Check environment variables."""
    import os

    print("\nChecking environment variables...")

    env_vars = [
        "OPENAI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ]

    missing = []
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var} - Set")
        else:
            print(f"  ⚠️  {var} - Not set")
            missing.append(var)

    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("Create a .env file with your API keys (see .env.example)")
        return False

    print("\n✓ Required environment variables set")
    return True


def check_project_structure():
    """Check project structure."""
    print("\nChecking project structure...")

    required_dirs = [
        "src/agents",
        "src/data",
        "src/evaluation",
        "src/retrieval",
        "src/rag",
        "src/config",
        "src/utils",
        "experiments/configs",
        "data",
    ]

    project_root = Path(__file__).parent.parent
    missing = []

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - NOT FOUND")
            missing.append(dir_path)

    if missing:
        print(f"\n⚠️  Missing directories: {', '.join(missing)}")
        return False

    print("\n✓ Project structure verified")
    return True


def test_configuration_loading():
    """Test loading experiment configuration."""
    print("\nTesting configuration loading...")

    try:
        from src.config.experiments import ExperimentConfig

        config_path = Path(__file__).parent.parent / "experiments/configs/dev.yaml"

        if not config_path.exists():
            print(f"  ✗ Config file not found: {config_path}")
            return False

        config = ExperimentConfig.from_yaml(config_path)
        print(f"  ✓ Loaded config: {config.name}")
        print(f"    - Run ID: {config.run_id}")
        print(f"    - Pipeline: {config.pipeline_type}")
        print(f"    - Retrieval: {config.retrieval_config.strategy}")

        print("\n✓ Configuration loading works")
        return True

    except Exception as e:
        print(f"  ✗ Error loading configuration: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("FinDER RAG System - Setup Validation")
    print("=" * 70)

    checks = [
        ("Package Imports", check_imports),
        ("Environment Variables", check_environment),
        ("Project Structure", check_project_structure),
        ("Configuration Loading", test_configuration_loading),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")

    if all_passed:
        print("\n✓ All checks passed! System ready for use.")
        print("\nNext steps:")
        print("  1. Download FinDER dataset: make download-data")
        print("  2. Run dev evaluation: make eval-dev")
        print("  3. Run full baseline: make eval-baseline")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
