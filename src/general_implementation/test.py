"""
End-to-end test runner for semantic_join.

Two examples:
  - imdb      : IMDB review sentiment match — exercises the classifier path.
  - numeric   : synthetic price-range match — exercises the pairwise path
                (semantic embeddings + clustering).

Usage:
    python test.py              # runs imdb
    python test.py numeric      # runs numeric
    python test.py --no-advisor # pin embedding/clustering explicitly
"""

import os
import pandas as pd
from dotenv import load_dotenv
from semantic_join import semantic_join

load_dotenv()
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def run_imdb(use_advisor: bool) -> None:
    table_a = pd.read_csv(os.path.join(DATA, "table_a.csv"))
    table_b = pd.read_csv(os.path.join(DATA, "table_b.csv"))

    result = semantic_join(
        table_a, table_b,
        predicate="both reviews express the same sentiment (both positive or both negative)",
        schema_a=["review"],
        schema_b=["review"],
        embedding=None if use_advisor else "all-MiniLM-L6-v2",
        clustering=None if use_advisor else "kmeans",
        n_clusters=5,
        filter_threshold=0.1,
        filter_sample_size=5,
        min_profile_size=4,
        block_size=15,
        cluster_size_limit=225,
    )
    print(result.summary(
        table_a, table_b,
        predicate_fn=lambda a, b: a["sentiment"] == b["sentiment"],
    ))


def main() -> None:
    run_imdb(use_advisor=True)


if __name__ == "__main__":
    main()
