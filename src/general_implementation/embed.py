"""
Row embeddings for the semantic join.

Rows are flattened to a single string and embedded with a sentence-transformer
model. The advisor picks the model from `prompts.EMBEDDING_MODELS`.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from utils import serialize_row


def embed(
    df: pd.DataFrame,
    schema: list[str],
    model_name: str | None = None,
    max_chars: int = 400,
) -> np.ndarray:
    """Embed rows using a sentence-transformer model.

    Returns an L2-normalized float32 array of shape (n_rows, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    texts = [serialize_row(row, schema, max_chars) for _, row in df.iterrows()]
    vecs = SentenceTransformer(model_name or "all-mpnet-base-v2").encode(
        texts, convert_to_numpy=True, batch_size=64
    )
    return normalize(vecs, norm="l2").astype(np.float32)
