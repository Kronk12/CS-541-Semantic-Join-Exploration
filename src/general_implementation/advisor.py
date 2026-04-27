"""
LLM-driven choices: join strategy, embedding model, and clustering method.

The advisor is only consulted when the user hasn't pinned a value. Each call
is independent — no global cache — since a full run makes at most three.
"""

import json
import math
import os
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

import prompts

load_dotenv()

_client: OpenAI | None = None


def _llm(system: str, user: str, model: str) -> dict:
    """Make an LLM call with JSON response format."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


@dataclass
class Plan:
    """Resolved join strategy, embedding model, clustering, and advisor notes."""
    join_strategy: str       # "classifier" | "pairwise"
    labels: list[str] | None # classifier label set (if join_strategy == classifier)
    model: str | None        # sentence-transformer name (pairwise only)
    clustering: str | None   # "kmeans" | "hdbscan"
    n_clusters: int | None   # kmeans
    min_cluster_size: int | None  # hdbscan
    notes: list[str]         # advisor rationales, for the run summary


def _samples(df: pd.DataFrame, schema: list[str], n: int = 5) -> list[dict]:
    """Extract first n rows from DataFrame as dicts for prompts."""
    return df[schema].head(n).to_dict(orient="records")


# def choose_join_strategy(
#     predicate: str,
#     df_a: pd.DataFrame, df_b: pd.DataFrame,
#     schema_a: list[str], schema_b: list[str],
#     llm_model: str,
# ) -> tuple[str, list[str] | None, str]:
#     """Decide whether the predicate is a same-label equi-join with a small,
#     fixed label set. Returns (strategy, labels_or_None, reason).
#     strategy is "classifier" or "pairwise"."""
#     system, user = prompts.classifier_detect_prompt(
#         predicate, schema_a, schema_b,
#         _samples(df_a, schema_a), _samples(df_b, schema_b),
#     )
#     result = _llm(system, user, llm_model)
#     is_classifier = bool(result.get("classifier"))
#     labels = result.get("labels")
#     reason = result.get("reason", "")

#     if not is_classifier or not isinstance(labels, list) or len(labels) < 2:
#         return "pairwise", None, reason

#     # Clean label set: dedup, keep strings, ensure "unknown" exists.
#     seen = set()
#     clean: list[str] = []
#     for label in labels:
#         if isinstance(label, str) and label not in seen:
#             seen.add(label)
#             clean.append(label)
#     if "unknown" not in seen:
#         clean.append("unknown")
#     return "classifier", clean, reason

def determine_join_strategy(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
) -> tuple[str, str]:
    """Call 1: Router. Decide whether the predicate is a same-label equi-join."""
    system, user = prompts.classifier_detect_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result = _llm(system, user, llm_model)
    is_classifier = bool(result.get("classifier"))
    reason = result.get("reason", "")

    if is_classifier:
        return "classifier", reason
    return "pairwise", reason


def generate_classification_labels(
    predicate: str,
    df_b: pd.DataFrame,
    schema_b: list[str],
    llm_model: str,
) -> list[str]:
    """Call 2: Extractor. Extracts the taxonomy strictly from Table B."""
    system = "You are an AI database query planner. Always respond in strictly valid JSON."
    user = f"""
    We are performing a classification join based on the predicate: "{predicate}"
    Look at this sample of the target table (Table B) with schema {schema_b}:
    {_samples(df_b, schema_b, n=25)}
    
    Identify the exact list of unique categories/labels we should classify Table A into.
    Return a JSON object with a single key "labels" mapped to a list of strings.
    """
    result = _llm(system, user, llm_model)
    labels = result.get("labels", [])
    
    # Clean label set: dedup, keep strings, ensure "unknown" exists.
    seen = set()
    clean: list[str] = []
    for label in labels:
        if isinstance(label, str) and label not in seen:
            seen.add(label)
            clean.append(label)
    if "unknown" not in seen:
        clean.append("unknown")
        
    return clean


def choose_model(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
) -> tuple[str, str]:
    """Pick sentence-transformer model from the catalog.
    Returns (model_name, reason)."""
    system, user = prompts.model_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result = _llm(system, user, llm_model)
    valid = {m["name"] for m in prompts.EMBEDDING_MODELS}
    model = result.get("model")
    if model not in valid:
        model = "all-mpnet-base-v2"
    return model, result.get("reason", "")


def choose_clustering(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    embedding_dim: int,
    llm_model: str,
) -> tuple[str, int | None, int | None, str]:
    """Pick clustering algorithm and hyperparameters.
    Returns (method, n_clusters or None, min_cluster_size or None, reason)."""
    n_rows_a, n_rows_b = len(df_a), len(df_b)
    max_k = max(2, int(math.sqrt(min(n_rows_a, n_rows_b))))
    system, user = prompts.clustering_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
        n_rows_a, n_rows_b, embedding_dim, max_k,
    )
    result = _llm(system, user, llm_model)
    method = result.get("method", "kmeans")
    if method not in {"kmeans", "hdbscan"}:
        method = "kmeans"

    n_clusters = result.get("n_clusters")
    min_cluster_size = result.get("min_cluster_size")

    if method == "kmeans":
        k = n_clusters or 5
        k = max(2, min(max_k, int(k)))
        return method, k, None, result.get("reason", "")
    else:
        m = min_cluster_size or 5
        m = max(2, int(m))
        return method, None, m, result.get("reason", "")

def choose_projection(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
) -> tuple[bool, str]:
    """Decide if project-sim-filter is needed instead of standard sim-filter.
    Returns (requires_projection, reason)."""
    system, user = prompts.projection_detect_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result = _llm(system, user, llm_model)
    
    requires_proj = bool(result.get("requires_projection", False))
    reason = result.get("reason", "")
    
    return requires_proj, reason