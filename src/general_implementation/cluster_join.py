"""
LLM-backed join of clustered row groups.

`join_clusters` walks every (cluster_a, cluster_b) pair, block-partitions
pairs that exceed `cluster_size_limit` rows of cross-product, and sends
each block-pair to the LLM with an inclusion prompt.
"""

import json
import os
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import prompts
from utils import TokenUsage, chunk_df, format_block, parse_id

load_dotenv()

_client: OpenAI | None = None

# Dataclass for cluster pair tracking
@dataclass
class ClusterPairStats:
    ca: int
    cb: int
    size_a: int
    size_b: int
    total_pairs: int
    matches: list[tuple[int, int]]
    tokens: TokenUsage
    n_llm_calls: int


def _llm(prompt: str, model: str) -> tuple[str, TokenUsage]:
    """Make an LLM call for join, return response text and token usage."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    # Reasoning models (o-series) reject temperature.
    if not any(tag in model for tag in ("o1", "o3", "o4")):
        kwargs["temperature"] = 0
    resp = _client.chat.completions.create(**kwargs)
    return (
        resp.choices[0].message.content,
        TokenUsage(prompt_tokens=resp.usage.prompt_tokens,
                   completion_tokens=resp.usage.completion_tokens),
    )


def _parse(raw: str, valid_pairs: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Parse LLM response JSON into match pairs, filtering invalid hallucinations."""
    out: list[tuple[int, int]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out
    for a_id, b_ids in (data.get("matches") or {}).items():
        try:
            a_idx = parse_id(a_id)
        except (ValueError, TypeError):
            continue
        for b_id in b_ids or []:
            try:
                pair = (a_idx, parse_id(b_id))
            except (ValueError, TypeError):
                continue
            if pair in valid_pairs:
                out.append(pair)
    return out


@dataclass
class JoinResult:
    matches: list[tuple[int, int]]
    tokens: TokenUsage
    n_llm_calls: int
    pair_stats: list[ClusterPairStats] = field(default_factory=list)


def _join_block_pair(
    block_a: pd.DataFrame, block_b: pd.DataFrame,
    predicate: str, schema_a: list[str], schema_b: list[str],
    llm_model: str, max_chars: int,
) -> tuple[list[tuple[int, int]], TokenUsage]:
    """Send one block-pair cross-product to LLM for matching."""
    valid = {(int(i), int(j)) for i in block_a.index for j in block_b.index}
    prompt = prompts.join_prompt(
        predicate,
        format_block(block_a, "A", schema_a, max_chars),
        format_block(block_b, "B", schema_b, max_chars),
    )
    raw, usage = _llm(prompt, llm_model)
    return _parse(raw, valid), usage


def join_cluster_pair(
    cluster_a: pd.DataFrame, cluster_b: pd.DataFrame,
    predicate: str, schema_a: list[str], schema_b: list[str],
    llm_model: str,
    block_size: int, cluster_size_limit: int, max_chars: int,
    verbose: bool = False,
) -> JoinResult:
    """Join one cluster pair, auto-partitioning if large.

    Strategy:
    - If the full cross-product fits under `cluster_size_limit`, one LLM call.
    - Else chunk only the larger side into `block_size` pieces and send each
      chunk against the full other side — every row-pair is evaluated exactly
      once, no cross-block gaps.
    - If even `block_size * len(other)` still exceeds the limit, fall back to
      chunking both sides (the legacy path).
    """
    matches: list[tuple[int, int]] = []
    tokens = TokenUsage()
    calls = 0

    # OVERRIDE: Enforce a strict 2D grid to prevent asymmetric hallucination.
    # This guarantees that if one side is chunked, the other side is either
    # already smaller than block_size, or it gets chunked too.
    if cluster_size_limit == -1:
        cluster_size_limit = block_size ** 2

    na, nb = len(cluster_a), len(cluster_b)

    if na * nb <= cluster_size_limit:
        m, u = _join_block_pair(
            cluster_a, cluster_b, predicate, schema_a, schema_b,
            llm_model, max_chars,
        )
        matches.extend(m)
        tokens += u
        calls = 1
        if verbose:
            print(f"  [join] {na}x{nb} → {len(m)} matches, {u.total:,} tok")
        return JoinResult(matches=matches, tokens=tokens, n_llm_calls=calls)

    # Chunk only the larger side; keep the other whole.
    if na >= nb:
        big, small = cluster_a, cluster_b
        big_is_a = True
    else:
        big, small = cluster_b, cluster_a
        big_is_a = False

    if block_size * len(small) <= cluster_size_limit:
        chunks = chunk_df(big, block_size)
        if verbose:
            print(f"  [join] {na}x{nb} → {len(chunks)} chunks of "
                  f"{'A' if big_is_a else 'B'} × full "
                  f"{'B' if big_is_a else 'A'}")
        for c in chunks:
            ba, bb = (c, small) if big_is_a else (small, c)
            m, u = _join_block_pair(
                ba, bb, predicate, schema_a, schema_b,
                llm_model, max_chars,
            )
            matches.extend(m)
            tokens += u
            calls += 1
    else:
        blocks_a = chunk_df(cluster_a, block_size)
        blocks_b = chunk_df(cluster_b, block_size)
        if verbose:
            print(f"  [join] {na}x{nb} → {len(blocks_a)}x{len(blocks_b)} "
                  f"blocks (both sides chunked)")
        for ba in blocks_a:
            for bb in blocks_b:
                m, u = _join_block_pair(
                    ba, bb, predicate, schema_a, schema_b,
                    llm_model, max_chars,
                )
                matches.extend(m)
                tokens += u
                calls += 1

    if verbose:
        print(f"  [join] {calls} calls → {len(matches)} matches, "
              f"{tokens.total:,} tok")
    return JoinResult(matches=matches, tokens=tokens, n_llm_calls=calls)


def join_clusters(
    table_a: pd.DataFrame, table_b: pd.DataFrame,
    labels_a, labels_b,
    cluster_pairs: list[tuple[int, int]],
    predicate: str, schema_a: list[str], schema_b: list[str],
    llm_model: str,
    block_size: int, cluster_size_limit: int, max_chars: int,
    verbose: bool = False,
) -> JoinResult:
    """Join all cluster pairs, skip noise and empty clusters."""
    """Join every cluster pair in `cluster_pairs`. Pairs touching the
    hdbscan noise label (-1) or an empty cluster are silently skipped."""
    a = table_a.assign(_cluster=labels_a)
    b = table_b.assign(_cluster=labels_b)

    all_matches: list[tuple[int, int]] = []
    tokens = TokenUsage()
    calls = 0

    # list to hold tracking data
    pair_stats: list[ClusterPairStats] = []

    for i, (ca, cb) in enumerate(cluster_pairs, 1):
        if ca < 0 or cb < 0:
            continue
        ga = a[a["_cluster"] == ca].drop(columns="_cluster")
        gb = b[b["_cluster"] == cb].drop(columns="_cluster")
        if len(ga) == 0 or len(gb) == 0:
            continue
        if verbose:
            print(f"[join] pair {i}/{len(cluster_pairs)}: "
                  f"A-c{ca}({len(ga)}) x B-c{cb}({len(gb)})")
        res = join_cluster_pair(
            ga, gb, predicate, schema_a, schema_b, llm_model,
            block_size, cluster_size_limit, max_chars, verbose=verbose,
        )
        all_matches.extend(res.matches)
        tokens += res.tokens
        calls += res.n_llm_calls
        pair_stats.append(ClusterPairStats(
            ca=ca, 
            cb=cb,
            size_a=len(ga), 
            size_b=len(gb),
            total_pairs=len(ga) * len(gb),
            matches=res.matches,
            tokens=res.tokens,
            n_llm_calls=res.n_llm_calls
        ))

    # Dedup while preserving first-seen order.
    deduped = list(dict.fromkeys(all_matches))
    return JoinResult(matches=deduped, tokens=tokens, n_llm_calls=calls, pair_stats=pair_stats)
