# Quick Start Guide

Get the FinDER Multi-Agent RAG System up and running in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- OpenAI API key (or Anthropic API key)
- (Optional) Langfuse account for observability

## Step 1: Install Dependencies

```bash
# Create virtual environment and install packages
make setup

# This will:
# - Create venv/ directory
# - Install all dependencies from requirements.txt
# - Takes ~5-10 minutes depending on internet speed
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY=sk-...
# - LANGFUSE_PUBLIC_KEY=pk-lf-... (optional)
# - LANGFUSE_SECRET_KEY=sk-lf-... (optional)
```

**Note**: If you don't have Langfuse credentials, the system will still work but won't log to Langfuse.

## Step 3: Validate Setup

```bash
# Run validation script
make validate

# Expected output:
# ✓ All required packages found
# ✓ Required environment variables set
# ✓ Project structure verified
# ✓ Configuration loading works
```

## Step 4: Download Dataset

```bash
# Download FinDER dataset from HuggingFace
make download-data

# This will:
# - Download 5,703 financial question-answer pairs
# - Save to data/finder/
# - Create metadata file with version info
# - Takes ~2-5 minutes
```

## Step 5: Run First Evaluation

### Option A: Quick Test (10 queries, ~30 seconds)

```bash
# Run on small subset for testing
make eval-dev

# Uses experiments/configs/dev.yaml
# Processes just 10 queries
# Good for verifying everything works
```

**Expected output**:
```
Starting Evaluation Run: dev-subset
Loading FinDER dataset...
Using subset: 10 queries
Processing 10 queries...
Evaluating queries: 100%|██████████| 10/10

Answer Correctness:
  Exact Match Rate: 0.200
  Mean Token F1:    0.450
  Semantic Sim:     0.680

Retrieval Quality:
  Precision:        0.400
  Recall:           0.350
  F1:               0.370
  MRR:              0.550

Operational:
  Mean Latency:     2.50s
  Total Cost:       $0.15

✓ Evaluation run complete: dev-subset
```

### Option B: Full Baseline Evaluation (5,703 queries, ~2-3 hours)

```bash
# Run full baseline evaluation
make eval-baseline

# Uses experiments/configs/baseline.yaml
# Processes all 5,703 queries
# Estimated cost: ~$2,000 (GPT-3.5-turbo)
# Estimated time: 2-3 hours (depends on API rate limits)
```

## What You'll Get

After running an evaluation, you'll have:

1. **Console Output**: Summary metrics printed to terminal
2. **Langfuse Traces** (if configured): Detailed logs at https://cloud.langfuse.com
3. **Metrics**: Answer accuracy, retrieval quality, latency, cost

## Common Issues

### Issue: "Package not found"

**Solution**: Make sure you ran `make setup` successfully
```bash
make setup
```

### Issue: "OpenAI API key not found"

**Solution**: Check your .env file has OPENAI_API_KEY set
```bash
cat .env  # Should show OPENAI_API_KEY=sk-...
```

### Issue: "FinDER dataset not found"

**Solution**: Download the dataset first
```bash
make download-data
```

### Issue: "Rate limit exceeded"

**Solution**: OpenAI has rate limits. The system will retry automatically, but you can:
- Wait a few minutes between runs
- Use a different API key
- Reduce subset size in dev.yaml

## Next Steps

### Explore Configurations

Edit experiment configurations in `experiments/configs/`:

```yaml
# experiments/configs/my-experiment.yaml
name: "My Custom Experiment"
description: "Testing with different settings"
run_id: "exp-custom-001"
pipeline_type: "baseline"

retrieval_config:
  strategy: "bm25"    # or "dense" or "hybrid"
  top_k: 10           # retrieve 10 passages instead of 5

llm_configs:
  generator:
    provider: "openai"
    model: "gpt-4"    # use GPT-4 instead of GPT-3.5
    temperature: 0.0
    max_tokens: 512
```

Run your custom config:
```bash
python -m src.evaluation.runner --config experiments/configs/my-experiment.yaml
```

### Try Different Retrieval Strategies

**BM25** (keyword-based):
```yaml
retrieval_config:
  strategy: "bm25"
  top_k: 5
```

**Dense** (semantic search):
```yaml
retrieval_config:
  strategy: "dense"
  top_k: 5
  embedding_model: "all-MiniLM-L6-v2"
```

**Hybrid** (best of both):
```yaml
retrieval_config:
  strategy: "hybrid"
  top_k: 5
  bm25_weight: 0.4
  dense_weight: 0.6
```

### View Results in Langfuse

If you configured Langfuse:

1. Go to https://cloud.langfuse.com
2. Find your run_id (e.g., "dev-subset" or "v0.0.0-baseline")
3. Explore individual query traces
4. See metrics and performance breakdown

## Help

For more information:
- **Full documentation**: See [README.md](README.md)
- **Implementation status**: See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Specification**: See [specs/001-finder-rag-system/spec.md](specs/001-finder-rag-system/spec.md)
- **Available commands**: Run `make help`

## Cost Warning ⚠️

Running evaluations costs money:
- **Dev subset (10 queries)**: ~$0.10-0.20
- **Full baseline (5,703 queries)**: ~$1,500-2,500 (GPT-3.5-turbo)

Always test with `make eval-dev` first before running full evaluations!
