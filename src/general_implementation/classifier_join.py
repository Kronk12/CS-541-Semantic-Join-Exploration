"""
Classifier-join: label each row with a fixed label, then equi-join on labels.

Used when the advisor decides the predicate is a "same-label" join
"""

import json
import os
import prompts
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

from utils import TokenUsage, chunk_df, format_block

load_dotenv()

_client: OpenAI | None = None


def _llm(system: str, user: str, model: str) -> tuple[str, TokenUsage]:
    """Make an LLM call, return response text and token usage."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    if not any(tag in model for tag in ("o1", "o3", "o4")):
        kwargs["temperature"] = 0
    resp = _client.chat.completions.create(**kwargs)
    return (
        resp.choices[0].message.content,
        TokenUsage(prompt_tokens=resp.usage.prompt_tokens,
                   completion_tokens=resp.usage.completion_tokens),
    )


def _label_batch(
    df: pd.DataFrame, prefix: str, schema: list[str],
    predicate: str, labels: list[str],
    llm_model: str, max_chars: int,
) -> tuple[dict[int, str], TokenUsage]:
    """Label one batch of rows; return {row_index: label} and token usage."""
    rows_text = format_block(df, prefix, schema, max_chars)
    system, user = prompts.classifier_label_prompt(predicate, labels, rows_text)
    raw, usage = _llm(system, user, llm_model)

    allowed = set(labels)
    out: dict[int, str] = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out, usage

    for row_id, label in (data.get("labels") or {}).items():
        if not isinstance(label, str) or label not in allowed:
            label = "unknown"
        # Row IDs look like "A-42" or "B-7"; strip the prefix.
        try:
            idx = int(str(row_id).split("-")[-1])
        except (ValueError, TypeError):
            continue
        out[idx] = label
    return out, usage


def _label_df(
    df: pd.DataFrame, prefix: str, schema: list[str],
    predicate: str, labels: list[str],
    llm_model: str, batch_size: int, max_chars: int,
    verbose: bool = False,
) -> tuple[dict[int, str], TokenUsage, int]:
    """Label every row in df by batching; return labels, tokens, n_calls."""
    result: dict[int, str] = {}
    tokens = TokenUsage()
    calls = 0
    for i, batch in enumerate(chunk_df(df, batch_size), 1):
        row_labels, usage = _label_batch(
            batch, prefix, schema, predicate, labels, llm_model, max_chars,
        )
        result.update(row_labels)
        tokens += usage
        calls += 1
        if verbose:
            print(f"  [label {prefix}] batch {i}: "
                  f"{len(row_labels)}/{len(batch)} labeled, {usage.total:,} tok")
    # Any row the LLM skipped gets "unknown".
    for idx in df.index:
        result.setdefault(int(idx), "unknown")
    return result, tokens, calls


@dataclass
class ClassifierJoinResult:
    matches: list[tuple[int, int]]
    tokens: TokenUsage
    n_llm_calls: int
    labels_a: dict[int, str]
    labels_b: dict[int, str]


def classifier_join(
    table_a: pd.DataFrame, table_b: pd.DataFrame,
    predicate: str, schema_a: list[str], schema_b: list[str],
    labels: list[str],
    llm_model: str,
    batch_size: int = 25,
    max_chars: int = 400,
    verbose: bool = False,
) -> ClassifierJoinResult:
    """Label every row in both tables, then equi-join on label.

    Rows labeled "unknown" never match anything. Returns every (a_idx, b_idx)
    pair whose labels agree.
    """
    if verbose:
        print(f"[classifier] labels={labels}")

    la, ta, ca = _label_df(
        table_a, "A", schema_a, predicate, labels,
        llm_model, batch_size, max_chars, verbose=verbose,
    )
    lb, tb, cb = _label_df(
        table_b, "B", schema_b, predicate, labels,
        llm_model, batch_size, max_chars, verbose=verbose,
    )
    tokens = ta
    tokens += tb

    # Bucket B indices by label, then walk A.
    b_by_label: dict[str, list[int]] = {}
    for idx, label in lb.items():
        if label == "unknown":
            continue
        b_by_label.setdefault(label, []).append(idx)

    matches: list[tuple[int, int]] = []
    for a_idx, a_label in la.items():
        if a_label == "unknown":
            continue
        for b_idx in b_by_label.get(a_label, ()):
            matches.append((a_idx, b_idx))

    if verbose:
        from collections import Counter
        print(f"[classifier] A labels: {dict(Counter(la.values()))}")
        print(f"[classifier] B labels: {dict(Counter(lb.values()))}")
        print(f"[classifier] {len(matches)} matches, {tokens.total:,} tok")

    return ClassifierJoinResult(
        matches=matches, tokens=tokens, n_llm_calls=ca + cb,
        labels_a=la, labels_b=lb,
    )
