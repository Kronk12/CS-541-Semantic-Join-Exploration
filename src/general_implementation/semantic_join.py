"""
semantic_join — join two tables on a natural-language predicate.

Two paths, both picked by an LLM advisor:
  * classifier  — predicate implies a small label set (sentiment, genre, …);
                  label every row once and equi-join on the labels.
  * pairwise    — embed rows with a sentence-transformer, cluster, optionally
                  sample-filter cluster pairs, then LLM-join surviving pairs.

Example
-------
    matches = semantic_join(
        table_a, table_b,
        predicate="both reviews express the same sentiment",
    )
"""

from __future__ import annotations

import itertools
import time
import pandas as pd
import advisor
import classifier_join
import cluster as _cluster
import embed as _embed
import cluster_filter
import cluster_join
from dataclasses import dataclass
from utils import TokenUsage, compute_metrics, ground_truth


@dataclass
class JoinResult:
    matches: pd.DataFrame            # columns: a_idx, b_idx
    tokens: TokenUsage
    timings: dict                    # stage name -> seconds
    plan: advisor.Plan

    def summary(
        self,
        table_a: pd.DataFrame | None = None,
        table_b: pd.DataFrame | None = None,
        predicate_fn=None,
    ) -> str:
        p = self.plan
        lines = [
            "=" * 60,
            "SEMANTIC JOIN",
            "=" * 60,
            f"  join       : {p.join_strategy}"
            + (f" labels={p.labels}" if p.labels else ""),
        ]
        if p.join_strategy == "pairwise":
            lines += [
                f"  embedding  : {p.model or 'all-mpnet-base-v2'}",
                f"  clustering : {p.clustering}"
                + (f" k={p.n_clusters}" if p.n_clusters else "")
                + (f" min={p.min_cluster_size}" if p.min_cluster_size else ""),
            ]
        lines += [
            f"  matches    : {len(self.matches)}",
            f"  tokens     : {self.tokens}",
            "  timings    : " + ", ".join(
                f"{k}={v:.1f}s" for k, v in self.timings.items()
            ),
        ]
        for note in p.notes:
            lines.append(f"  advisor    : {note}")

        if predicate_fn is not None and table_a is not None and table_b is not None:
            gt = ground_truth(table_a, table_b, predicate_fn)
            pred = set(zip(self.matches["a_idx"], self.matches["b_idx"]))
            m = compute_metrics(gt, pred)
            lines += ["-" * 60, f"  eval       : {m} ({len(gt)} true matches)"]

        lines.append("=" * 60)
        return "\n".join(lines)


def _resolve_model(
    embedding: str | None,
    predicate: str,
    table_a, table_b, schema_a, schema_b, llm_model,
) -> tuple[str, list[str]]:
    """Return (sentence_transformer_model_name, notes).

    If the user passed an explicit model name, use it; otherwise ask the advisor.
    """
    notes: list[str] = []
    if embedding is not None:
        return embedding, notes

    model, reason = advisor.choose_model(
        predicate, table_a, table_b, schema_a, schema_b, llm_model,
    )
    notes.append(f"model={model} — {reason}")
    return model, notes


def _resolve_clustering(
    clustering: str | None,
    predicate: str,
    table_a, table_b, schema_a, schema_b, embedding_dim: int,
    n_clusters: int | None, min_cluster_size: int | None,
    llm_model: str,
) -> tuple[str, int | None, int | None, list[str]]:
    notes: list[str] = []

    if clustering is None:
        method, k, m, reason = advisor.choose_clustering(
            predicate, table_a, table_b, schema_a, schema_b,
            embedding_dim, llm_model,
        )
        notes.append(f"clustering={method} — {reason}")
        return method, k, m, notes

    if clustering == "kmeans":
        return "kmeans", n_clusters or 5, None, notes
    if clustering == "hdbscan":
        return "hdbscan", None, min_cluster_size or 5, notes
    raise ValueError(f"clustering must be 'kmeans', 'hdbscan', or None — got {clustering!r}")


def _cluster_pairs(labels_a, labels_b) -> list[tuple[int, int]]:
    a = sorted(c for c in set(int(x) for x in labels_a) if c >= 0)
    b = sorted(c for c in set(int(x) for x in labels_b) if c >= 0)
    return list(itertools.product(a, b))


def semantic_join(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    predicate: str,
    schema_a: list[str] | None = None,
    schema_b: list[str] | None = None,
    llm_model: str = "gpt-4o",
    embedding: str | None = None,
    clustering: str | None = None,
    n_clusters: int | None = None,
    min_cluster_size: int | None = None,
    filter_threshold: float = 0.1,
    filter_sample_size: int = 10,
    min_profile_size: int = 40,
    block_size: int = 20,
    cluster_size_limit: int = 1000,
    max_chars_per_col: int = 400,
    random_state: int = 42,
    verbose: bool = True,
) -> JoinResult:
    """Join `table_a` and `table_b` on a natural-language `predicate`.

    `schema_a` / `schema_b` default to every column in the respective table.

    `embedding` is `None` to let the advisor pick a sentence-transformer
    model from the catalog, or an explicit model name (e.g. "all-MiniLM-L6-v2").
    `clustering` is `None` (advisor picks), "kmeans", or "hdbscan".

    `filter_threshold=0` disables the sample-based filter.
    """
    if schema_a is None:
        schema_a = list(table_a.columns)
    if schema_b is None:
        schema_b = list(table_b.columns)
    for col in schema_a:
        if col not in table_a.columns:
            raise ValueError(f"schema_a column {col!r} not in table_a")
    for col in schema_b:
        if col not in table_b.columns:
            raise ValueError(f"schema_b column {col!r} not in table_b")

    timings: dict[str, float] = {}
    plan_notes: list[str] = []
    total_tokens = TokenUsage()

    # Stage 0: pick join strategy (classifier vs pairwise). Skipped when
    # specified.
    t0 = time.time()
    if embedding is None and clustering is None:
        jstrat, labels, reason = advisor.choose_join_strategy(
            predicate, table_a, table_b, schema_a, schema_b, llm_model,
        )
        plan_notes.append(f"join_strategy={jstrat} — {reason}")
    else:
        jstrat, labels = "pairwise", None
    timings["advise"] = time.time() - t0
    if verbose:
        print(f"[advise] join_strategy={jstrat}"
              + (f" labels={labels}" if labels else ""))

    if jstrat == "classifier":
        t0 = time.time()
        cjr = classifier_join.classifier_join(
            table_a, table_b, predicate, schema_a, schema_b, labels,
            llm_model, max_chars=max_chars_per_col, verbose=verbose,
        )
        total_tokens += cjr.tokens
        timings["classifier_join"] = time.time() - t0
        matches_df = pd.DataFrame(cjr.matches, columns=["a_idx", "b_idx"])
        plan = advisor.Plan(
            join_strategy="classifier", labels=labels,
            model=None,
            clustering=None, n_clusters=None, min_cluster_size=None,
            notes=plan_notes,
        )
        result = JoinResult(
            matches=matches_df, tokens=total_tokens,
            timings=timings, plan=plan,
        )
        if verbose:
            print()
            print(result.summary())
        return result

    # Stage 1: pick sentence-transformer model and embed both tables.
    t0 = time.time()
    model, notes = _resolve_model(
        embedding, predicate, table_a, table_b, schema_a, schema_b, llm_model,
    )
    plan_notes.extend(notes)
    if verbose:
        print(f"[embed] model={model}")

    emb_a = _embed.embed(table_a, schema_a, model, max_chars_per_col)
    emb_b = _embed.embed(table_b, schema_b, model, max_chars_per_col)
    timings["embed"] = time.time() - t0
    if verbose:
        print(f"[embed] A={emb_a.shape} B={emb_b.shape} ({timings['embed']:.1f}s)")

    # Stage 2: pick clustering + run.
    t0 = time.time()
    method, k, m, notes = _resolve_clustering(
        clustering, predicate,
        table_a, table_b, schema_a, schema_b, emb_a.shape[1],
        n_clusters, min_cluster_size, llm_model,
    )
    plan_notes.extend(notes)
    if verbose:
        print(f"[cluster] method={method}"
              + (f" k={k}" if k else "")
              + (f" min_cluster_size={m}" if m else ""))

    labels_a = _cluster.cluster(emb_a, method, n_clusters=k, min_cluster_size=m,
                                random_state=random_state)
    labels_b = _cluster.cluster(emb_b, method, n_clusters=k, min_cluster_size=m,
                                random_state=random_state)
    timings["cluster"] = time.time() - t0
    if verbose:
        print(f"[cluster] A: {_cluster.distribution(labels_a)}")
        print(f"[cluster] B: {_cluster.distribution(labels_b)}")

    pairs = _cluster_pairs(labels_a, labels_b)
    if verbose:
        print(f"[pipeline] {len(pairs)} candidate cluster pairs")

    # Stage 3: optional sample-based filter.
    t0 = time.time()
    if filter_threshold > 0 and pairs:
        outcome = cluster_filter.filter_clusters(
            table_a, table_b, labels_a, labels_b, pairs,
            predicate, schema_a, schema_b, llm_model,
            threshold=filter_threshold,
            sample_size=filter_sample_size,
            min_profile_size=min_profile_size,
            max_chars=max_chars_per_col,
            random_state=random_state,
            verbose=verbose,
        )
        surviving = outcome.kept
        total_tokens += outcome.tokens
        if verbose:
            print(f"[filter] kept {len(surviving)}/{len(pairs)} pairs, "
                  f"{outcome.tokens.total:,} tok")
    else:
        surviving = pairs
    timings["filter"] = time.time() - t0

    # Stage 4: LLM join.
    t0 = time.time()
    jr = cluster_join.join_clusters(
        table_a, table_b, labels_a, labels_b, surviving,
        predicate, schema_a, schema_b, llm_model,
        block_size=block_size,
        cluster_size_limit=cluster_size_limit,
        max_chars=max_chars_per_col,
        verbose=verbose,
    )
    total_tokens += jr.tokens
    timings["join"] = time.time() - t0

    matches_df = pd.DataFrame(jr.matches, columns=["a_idx", "b_idx"])

    plan = advisor.Plan(
        join_strategy="pairwise", labels=None,
        model=model,
        clustering=method, n_clusters=k, min_cluster_size=m,
        notes=plan_notes,
    )

    result = JoinResult(
        matches=matches_df, tokens=total_tokens,
        timings=timings, plan=plan,
    )

    if verbose:
        print()
        print(result.summary())

    return result
