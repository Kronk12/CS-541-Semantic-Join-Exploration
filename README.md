# Semantic Join

This repository contains the implementation and evaluation code for a semantic join system that uses LLMs to join two relational tables on a natural-language predicate (e.g., "both reviews express the same sentiment") without requiring a traditional equi-join key.

## Overview

The system works in two stages:

1. **LLM Advisor** — An LLM analyzes the tables and predicate to choose a join strategy:
   - **Classifier join**: Extract a label taxonomy, classify every row, then equi-join on label.
   - **Pairwise join**: Embed rows with a sentence-transformer, cluster each table, prune cluster pairs via sample-based filtering, then run an LLM join within surviving pairs.

2. **Optional projection** — For cross-schema joins, the LLM can rewrite Table A rows into Table B's domain before embedding.

## Repository Structure

```
data/                         # Benchmark datasets (IMDb, emails, Stack Overflow)
src/
├── general_implementation/   # Core join engine
│   ├── semantic_join.py      # Main orchestrator
│   ├── advisor.py            # LLM strategy routing
│   ├── embed.py              # Sentence-transformer embeddings
│   ├── cluster.py            # K-Means / HDBSCAN clustering
│   ├── cluster_filter.py     # Sample-based cluster-pair pruning
│   ├── cluster_join.py       # LLM join within cluster pairs
│   ├── classifier_join.py    # Label + equi-join path
│   ├── project.py            # LLM projection across schemas
│   ├── prompts.py            # All prompt templates
│   ├── simulate.py           # Export/replay simulation logs
│   ├── utils.py              # Helpers (token accounting, metrics)
│   └── test.py               # Quick smoke test
├── evaluation/               # Benchmark runners and baselines
├── preprocessing/            # Dataset generation / preparation
├── figures/                  # Visualization scripts and output PNGs
└── results/                  # Aggregated CSVs and simulation logs
```

## Datasets

| Dataset | Size | Join Predicate |
|---------|------|----------------|
| **IMDb Reviews** | 50 × 50 | Same sentiment (positive/negative) |
| **Emails** | 100 × 100 | Email contradicts witness statement |
| **Stack Overflow** | 100 × 10 | Question relates to programming concept |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=your-key-here
```

## Usage

<!-- **Quick test** (runs an IMDB semantic join end-to-end):

```bash
python src/general_implementation/test.py
``` -->

**Running a semantic join** (on the IMDb data set):

```python
from semantic_join import semantic_join

result = semantic_join(table_a, table_b,
                       predicate="both reviews express the same sentiment",
                       schema_a=["review"], schema_b=["review"])

matches = result.matches  # DataFrame with columns: a_idx, b_idx
```

**Evaluation scripts** (run from repo root):

```bash
python src/evaluation/evaluate_imdb.py
python src/evaluation/evaluate_emails.py
python src/evaluation/evaluate_stack_overflow.py
python src/evaluation/evaluate_naive.py          # naive baseline
python src/evaluation/evaluate_block.py          # full block baseline
python src/evaluation/evaluate_simulations.py    # threshold sweep on saved logs (no API calls)
```

**Figures** (generated from result CSVs):

```bash
python src/figures/ratio_visualization.py
python src/figures/threshold_visualization.py
python src/figures/block_size_visualization.py
```