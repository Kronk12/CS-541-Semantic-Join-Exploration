"""
Sample-based cluster-pair filter.

For each candidate cluster pair, draw a small random sample from each side,
run a mini-join on the sample via `join_cluster_pair`, and compute

    match_rate = n_matches / (|sample_a| * |sample_b|)

Keep the pair if match_rate >= threshold, drop otherwise.

When either cluster already fits under `min_profile_size`, profiling would
cost as much as just doing the join — so the pair is kept without an LLM
call and left for the join stage to handle directly.
"""

from dataclasses import dataclass

import pandas as pd

from cluster_join import join_cluster_pair
from utils import TokenUsage, sample_df


@dataclass
class FilterOutcome:
    kept: list[tuple[int, int]]
    dropped: list[tuple[int, int]]
    match_rates: dict[tuple[int, int], float]
    tokens: TokenUsage


def filter_clusters(
    table_a: pd.DataFrame, table_b: pd.DataFrame,
    labels_a, labels_b,
    cluster_pairs: list[tuple[int, int]],
    predicate: str, schema_a: list[str], schema_b: list[str],
    llm_model: str,
    threshold: float,
    sample_size: int,
    min_profile_size: int,
    max_chars: int,
    random_state: int = 42,
    verbose: bool = False,
) -> FilterOutcome:
    """Sample-profile cluster pairs, keep only those above match-rate threshold."""
    a = table_a.assign(_cluster=labels_a)
    b = table_b.assign(_cluster=labels_b)

    kept: list[tuple[int, int]] = []
    dropped: list[tuple[int, int]] = []
    rates: dict[tuple[int, int], float] = {}
    tokens = TokenUsage()

    for i, (ca, cb) in enumerate(cluster_pairs, 1):
        # Always drop noise-cluster pairs — they mean "unassigned".
        if ca < 0 or cb < 0:
            dropped.append((ca, cb))
            continue

        ga = a[a["_cluster"] == ca].drop(columns="_cluster")
        gb = b[b["_cluster"] == cb].drop(columns="_cluster")
        if len(ga) == 0 or len(gb) == 0:
            dropped.append((ca, cb))
            continue

        # Too small to profile cheaply — keep and let the join stage handle it.
        if len(ga) <= min_profile_size or len(gb) <= min_profile_size:
            kept.append((ca, cb))
            if verbose:
                print(f"[filter] pair {i}/{len(cluster_pairs)} "
                      f"A-c{ca}({len(ga)}) x B-c{cb}({len(gb)}): "
                      f"below min_profile_size, kept")
            continue

        sa = sample_df(ga, sample_size, random_state=random_state)
        sb = sample_df(gb, sample_size, random_state=random_state + 1)

        # Make cluster_size_limit big enough that the sample fits in one LLM call.
        limit = len(sa) * len(sb) + 1
        res = join_cluster_pair(
            sa, sb, predicate, schema_a, schema_b, llm_model,
            block_size=max(len(sa), len(sb)),
            cluster_size_limit=limit,
            max_chars=max_chars,
            verbose=False,
        )
        tokens += res.tokens

        n_pairs = len(sa) * len(sb)
        rate = len(res.matches) / n_pairs if n_pairs else 0.0
        rates[(ca, cb)] = rate

        verdict = "keep" if rate >= threshold else "drop"
        (kept if rate >= threshold else dropped).append((ca, cb))

        if verbose:
            print(f"[filter] pair {i}/{len(cluster_pairs)} "
                  f"A-c{ca}({len(ga)}) x B-c{cb}({len(gb)}): "
                  f"rate={rate:.2f} {verdict}")

    return FilterOutcome(kept=kept, dropped=dropped, match_rates=rates, tokens=tokens)
