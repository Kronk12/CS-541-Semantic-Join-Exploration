"""
All LLM prompt templates live here so they can be tuned in one place.

Every template returns (system, user) and the response format is always JSON.
"""

import json


# Curated catalogs the advisor chooses from. Keeping them short makes the
# advisor's job easier and the prompts cheaper.

EMBEDDING_MODELS = [
    {
        "name": "all-mpnet-base-v2",
        "notes": "Strong general-purpose semantic similarity. Default if unsure.",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "notes": "Small and fast. Good when the signal is clear and speed matters.",
    },
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "notes": "Use only if the text is non-English or multilingual.",
    },
    {
        "name": "multi-qa-mpnet-base-dot-v1",
        "notes": "Tuned for question/answer retrieval, not general similarity.",
    },
]

CLUSTERING_METHODS = [
    {
        "name": "kmeans",
        "notes": "Pick when clusters are roughly balanced and k can be estimated.",
        "params": ["n_clusters"],
    },
    {
        "name": "hdbscan",
        "notes": "Pick when cluster count is unknown, sizes vary, or noise is likely.",
        "params": ["min_cluster_size"],
    },
]


def _sample_block(schema: list[str], rows: list[dict], limit: int = 5) -> str:
    """Format sample rows as a readable block for advisor prompts."""
    lines = []
    for i, row in enumerate(rows[:limit], 1):
        parts = [f"{c}={str(row.get(c, ''))[:120]}" for c in schema]
        lines.append(f"  {i}. " + " | ".join(parts))
    return "\n".join(lines) or "  (no rows)"


def model_prompt(
    predicate: str,
    schema_a: list[str], schema_b: list[str],
    samples_a: list[dict], samples_b: list[dict],
) -> tuple[str, str]:
    """Ask LLM to pick a sentence-transformer model from the catalog.
    Returns (system_prompt, user_prompt) tuple."""
    system = (
        "You pick a sentence-transformer model for a two-table semantic join. "
        "Respond with JSON only."
    )
    user = f"""Pick the best sentence-transformer model.

Predicate: "{predicate}"

Table A sample:
{_sample_block(schema_a, samples_a)}

Table B sample:
{_sample_block(schema_b, samples_b)}

Available models:
{json.dumps(EMBEDDING_MODELS, indent=2)}

Respond exactly:
{{"model": "<name from list>", "reason": "<one sentence>"}}
"""
    return system, user


def clustering_prompt(
    predicate: str,
    schema_a: list[str], schema_b: list[str],
    samples_a: list[dict], samples_b: list[dict],
    n_rows_a: int, n_rows_b: int, embedding_dim: int,
    max_k: int,
) -> tuple[str, str]:
    """Ask LLM to pick clustering algorithm and hyperparameters.
    Returns (system_prompt, user_prompt) tuple."""
    system = (
        "You pick a clustering algorithm for a two-table join. Respond with JSON only."
    )
    user = f"""Pick the best clustering algorithm.

Predicate: "{predicate}"
Table A: {n_rows_a} rows | Table B: {n_rows_b} rows | embedding dim: {embedding_dim}

Table A sample:
{_sample_block(schema_a, samples_a)}

Table B sample:
{_sample_block(schema_b, samples_b)}

Available methods:
{json.dumps(CLUSTERING_METHODS, indent=2)}

Think about what groups the predicate itself implies. If the predicate splits
rows into a small, known set of classes (e.g. "same sentiment" → 2, "same
genre" → ~5), prefer kmeans with n_clusters equal to that count. Only use
hdbscan when the predicate does not imply a fixed number of groups AND the
tables are large enough (>100 rows) for density-based clustering to be stable
— on small datasets hdbscan tends to mark everything as noise.

Rules:
- kmeans: pick n_clusters between 2 and {max_k}.
- hdbscan: pick min_cluster_size >= 2. Leaves "n_clusters" as null.

Respond exactly:
{{"method": "kmeans" | "hdbscan",
  "n_clusters": <int or null>,
  "min_cluster_size": <int or null>,
  "reason": "<one sentence>"}}
"""
    return system, user


def classifier_detect_prompt(
    predicate: str,
    schema_a: list[str], schema_b: list[str],
    samples_a: list[dict], samples_b: list[dict],
) -> tuple[str, str]:
    """Ask LLM whether the predicate is a 'same-label' join over a small, fixed
    set of classes — and if so, propose the label set.
    Returns (system_prompt, user_prompt) tuple."""
    system = (
        "You decide whether a two-table join predicate reduces to a same-label "
        "equi-join over a small, known set of classes. Respond with JSON only."
    )
    user = f"""Decide whether this predicate is a same-label join.

Predicate: "{predicate}"

Table A sample:
{_sample_block(schema_a, samples_a)}

Table B sample:
{_sample_block(schema_b, samples_b)}

A predicate is a "same-label" join when:
- It asks whether two rows share the same value of some discrete attribute
  (sentiment, genre, priority, department, product category, …), AND
- That attribute has a small, finite label set (2–10 labels) that is the same
  for both tables, AND
- Each row can be confidently labeled from its own content alone.

It is NOT a same-label join when:
- The predicate is about numeric proximity, continuous values, or ranges.
- The predicate needs cross-row comparison (e.g. "A happened before B").
- The label space is open-ended or unbounded (e.g. "same topic" with no fixed list).

If it IS a same-label join, propose an explicit, mutually-exclusive label set
that would cover every realistic row in both tables. Include a special
"unknown" label for rows that don't fit.

Respond exactly:
{{"classifier": true | false,
  "labels": ["<label1>", "<label2>", ...] | null,
  "reason": "<one sentence>"}}
"""
    return system, user


def classifier_label_prompt(
    predicate: str,
    labels: list[str],
    rows_text: str,
) -> tuple[str, str]:
    """Ask LLM to assign one label from `labels` to each row.
    Returns (system_prompt, user_prompt) tuple."""
    system = (
        "You label rows for a same-label equi-join. Pick exactly one label per "
        "row from the provided set. Respond with JSON only."
    )
    user = f"""Label each row with exactly one label from this set.

Predicate context: "{predicate}"

Labels: {json.dumps(labels)}

Rules:
- Pick exactly one label per row, chosen from the set above.
- Commit to your best-guess label even when the row is mixed or subtle —
  "unknown" is only for rows with no interpretable content (empty, gibberish,
  or off-topic). Do not use "unknown" to hedge on hard cases.
- Do not invent labels.

Rows:
{rows_text}

Respond exactly:
{{"labels": {{"<ID>": "<label>", ...}}}}
"""
    return system, user


def join_prompt(
    predicate: str,
    block_a_text: str, block_b_text: str,
) -> str:
    """Build the LLM prompt for joining one block pair.
    Asks LLM to return matching pairs in JSON format."""
    return f"""Find every pair of rows where: {predicate}

Compare every row in TABLE A against every row in TABLE B.

TABLE A:
{block_a_text}

TABLE B:
{block_b_text}

Return a JSON object mapping each TABLE A id to the list of matching TABLE B ids.
A row with no matches gets an empty list. No explanations.

Format:
{{"matches": {{"A-0": ["B-2", "B-5"], "A-1": [], "A-2": ["B-0"]}}}}
"""
