"""
LLM-based row projection (Project-Sim-Filter).
Projects rows from Table A into the domain of Table B prior to embedding.
"""

import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

import prompts
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
                   completion_tokens=resp.usage.completion_tokens)
    )

def _project_batch(
    df: pd.DataFrame, prefix: str, schema_a: list[str], schema_b: list[str],
    target_samples_text: str, predicate: str, llm_model: str, max_chars: int
) -> tuple[dict[int, str], TokenUsage]:
    rows_text = format_block(df, prefix, schema_a, max_chars)
    system, user = prompts.project_batch_prompt(predicate, schema_b, target_samples_text, rows_text)
    raw, usage = _llm(system, user, llm_model)

    out: dict[int, str] = {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return out, usage

    for row_id, proj_text in (data.get("projections") or {}).items():
        try:
            idx = int(str(row_id).split("-")[-1])
            out[idx] = str(proj_text)
        except (ValueError, TypeError):
            continue
    return out, usage

def project_df(
    df: pd.DataFrame, prefix: str, schema_a: list[str], schema_b: list[str],
    target_samples_text: str, predicate: str, llm_model: str, batch_size: int = 25,
    max_chars: int = 400, verbose: bool = False
) -> tuple[pd.Series, TokenUsage]:
    """Projects a dataframe using the LLM. Returns a Series of projected strings."""
    result: dict[int, str] = {}
    tokens = TokenUsage()
    
    for i, batch in enumerate(chunk_df(df, batch_size), 1):
        projs, usage = _project_batch(
            batch, prefix, schema_a, schema_b, target_samples_text, predicate, llm_model, max_chars
        )
        result.update(projs)
        tokens += usage
        if verbose:
            print(f"  [project {prefix}] batch {i}: "
                  f"{len(projs)}/{len(batch)} projected, {usage.total:,} tok")
            
    # Map back to a pandas Series, defaulting to an empty string if a row failed
    series = pd.Series([result.get(idx, "") for idx in df.index], index=df.index)
    return series, tokens