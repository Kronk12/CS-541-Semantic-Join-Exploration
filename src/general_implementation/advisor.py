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
from utils import TokenUsage

import prompts

load_dotenv()

_client: OpenAI | None = None


def _llm(system: str, user: str, model: str) -> tuple[dict, TokenUsage]:
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

    usage = resp.usage
    tokens = TokenUsage(
        prompt_tokens=usage.prompt_tokens, 
        completion_tokens=usage.completion_tokens, 
    )

    return json.loads(resp.choices[0].message.content), tokens


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


def _samples(df: pd.DataFrame, schema: list[str], n: int = 5, random_state: int = 42) -> list[dict]:
    """Extract a representative random sample of n rows from DataFrame as dicts for prompts."""
    sample_size = min(n, len(df))
    if sample_size == 0:
        return []
    return df[schema].sample(n=sample_size, random_state=random_state).to_dict(orient="records")

def determine_join_strategy(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
    return_tokens: bool = False,
    return_raw: bool = False
) -> tuple[str, str]:
    """Call 1: Router. Decide whether the predicate is a same-label equi-join."""
    system, user = prompts.classifier_detect_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result, tokens = _llm(system, user, llm_model)
    
    strategy = result.get("strategy", "unknown")
    if strategy not in {"classifier", "pairwise", "unknown"}:
        strategy = "unknown"

    reason = result.get("reason", "")

    if return_tokens and return_raw:
        return strategy, reason, tokens, json.dumps(result)
    elif return_tokens:
        return strategy, reason, tokens
    elif return_raw:
        return strategy, reason, json.dumps(result)
    return strategy, reason


def generate_classification_labels(
    predicate: str,
    df_b: pd.DataFrame,
    schema_b: list[str],
    llm_model: str,
    return_tokens: bool = False,
    return_raw: bool = False
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
    result, tokens = _llm(system, user, llm_model)
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
        
    if return_tokens and return_raw:
        return clean, tokens, json.dumps(result)
    elif return_tokens:
        return clean, tokens
    elif return_raw:
        return clean, json.dumps(result)
        
    return clean


def choose_model(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
    return_tokens: bool = False,
    return_raw: bool = False
) -> tuple:
    """Pick sentence-transformer model from the catalog.
    Returns (model_name, reason)."""
    system, user = prompts.model_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result, tokens = _llm(system, user, llm_model)
    valid = {m["name"] for m in prompts.EMBEDDING_MODELS}
    model = result.get("model")
    if model not in valid:
        model = "all-mpnet-base-v2"

    reason = result.get("reason", "")

    if return_tokens and return_raw:
        return model, reason, tokens, json.dumps(result)
    elif return_tokens:
        return model, reason, tokens
    elif return_raw:
        return model, reason, json.dumps(result)
    
    return model, reason


def choose_clustering(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    embedding_dim: int,
    llm_model: str,
    return_tokens: bool = False,
    return_raw: bool = False
) -> tuple:
    """Pick clustering algorithm and hyperparameters.
    Returns (method, n_clusters or None, min_cluster_size or None, reason)."""
    n_rows_a, n_rows_b = len(df_a), len(df_b)
    max_k = max(2, int(math.sqrt(min(n_rows_a, n_rows_b))))
    system, user = prompts.clustering_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
        n_rows_a, n_rows_b, embedding_dim, max_k,
    )
    result, tokens = _llm(system, user, llm_model)
    method = result.get("method", "kmeans")
    if method not in {"kmeans", "hdbscan"}:
        method = "kmeans"

    n_clusters = result.get("n_clusters")
    min_cluster_size = result.get("min_cluster_size")
    reason = result.get("reason", "")

    if method == "kmeans":
        k = n_clusters or 5
        k = max(2, min(max_k, int(k)))
        if return_tokens and return_raw:
            return method, k, None, reason, tokens, json.dumps(result)
        elif return_tokens:
            return method, k, None, reason, tokens
        elif return_raw:
            return method, k, None, reason, json.dumps(result)
        return method, k, None, reason
    else:
        m = min_cluster_size or 5
        m = max(2, int(m))
        if return_tokens and return_raw:
            return method, None, m, reason, tokens, json.dumps(result)
        elif return_tokens:
            return method, None, m, reason, tokens
        elif return_raw:
            return method, None, m, reason, json.dumps(result)
        return method, None, m, reason

def choose_projection(
    predicate: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    schema_a: list[str], schema_b: list[str],
    llm_model: str,
    return_tokens: bool = False,
    return_raw: bool = False
) -> tuple[bool, str]:
    """Decide if project-sim-filter is needed instead of standard sim-filter.
    Returns (requires_projection, reason)."""
    system, user = prompts.projection_detect_prompt(
        predicate, schema_a, schema_b,
        _samples(df_a, schema_a), _samples(df_b, schema_b),
    )
    result, tokens = _llm(system, user, llm_model)

    proj_val = str(result.get("requires_projection", "unknown")).lower()
    if proj_val not in {"true", "false", "unknown"}:
        proj_val = "unknown"
    
    reason = result.get("reason", "")
    
    if return_tokens and return_raw:
        return proj_val, reason, tokens, json.dumps(result)
    elif return_tokens:
        return proj_val, reason, tokens
    elif return_raw:
        return proj_val, reason, json.dumps(result)
    return proj_val, reason