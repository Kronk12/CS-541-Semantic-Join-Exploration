"""
Shared helpers: row serialization, ID parsing, token bookkeeping, metrics.
"""

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


_ID_PATTERN = re.compile(r"^[A-Za-z]+-(\d+)$")

def make_id(prefix: str, idx: int | str) -> str:
    """Create a labelled row ID like 'A-42'."""
    return f"{prefix}-{idx}"


def parse_id(val: str | int) -> int:
    """Extract row index from a labelled ID like 'A-42' or plain integer."""
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    m = _ID_PATTERN.match(s)
    return int(m.group(1)) if m else int(s)


def ids_to_pair(id_a: str | int, id_b: str | int) -> tuple[int, int]:
    """Parse two IDs and return them as an (int, int) pair."""
    return (parse_id(id_a), parse_id(id_b))


def serialize_row(row: pd.Series, schema: list[str], max_chars: int = 400) -> str:
    """Flatten a row to a string, truncating each column to max_chars."""
    parts = []
    for col in schema:
        value = str(row[col]) if col in row.index else ""
        if len(value) > max_chars:
            value = value[:max_chars] + "…"
        parts.append(f"{col}: {value}")
    return "|".join(parts)


def format_block(df: pd.DataFrame, prefix: str, schema: list[str],
                 max_chars: int = 400) -> str:
    """Format rows as labelled lines for LLM prompts."""
    return "\n".join(
        f"ID {make_id(prefix, idx)}: {serialize_row(row, schema, max_chars)}"
        for idx, row in df.iterrows()
    )


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self

    def __str__(self) -> str:
        return (
            f"prompt={self.prompt_tokens:,} "
            f"completion={self.completion_tokens:,} "
            f"total={self.total:,}"
        )


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    recall: float      # percent
    precision: float   # percent
    f1: float          # percent

    def __str__(self) -> str:
        return (f"TP={self.tp} FP={self.fp} FN={self.fn} | "
                f"R={self.recall:.1f}% P={self.precision:.1f}% F1={self.f1:.1f}%")


def compute_metrics(
    ground_truth: Iterable[tuple[int, int]],
    predicted: Iterable[tuple[int, int]],
) -> Metrics:
    """Compute precision, recall, F1 from ground truth and predicted match pairs."""
    gt = {(int(a), int(b)) for a, b in ground_truth}
    pred = {(int(a), int(b)) for a, b in predicted}

    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)

    recall = (tp / len(gt) * 100) if gt else 0.0
    precision = (tp / len(pred) * 100) if pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return Metrics(tp=tp, fp=fp, fn=fn, recall=recall, precision=precision, f1=f1)


def ground_truth(
    table_a: pd.DataFrame, table_b: pd.DataFrame, predicate_fn,
) -> set[tuple[int, int]]:
    """Compute ground truth by testing every pair against the predicate."""
    return {
        (int(i), int(j))
        for i, ra in table_a.iterrows()
        for j, rb in table_b.iterrows()
        if predicate_fn(ra, rb)
    }


def sample_df(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Sample up to n rows from the DataFrame."""
    return df.sample(n=min(n, len(df)), random_state=random_state)


def chunk_df(df: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    """Split DataFrame into blocks of at most 'size' rows each."""
    return [df.iloc[i : i + size] for i in range(0, len(df), size)]
