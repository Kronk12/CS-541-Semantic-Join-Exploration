import sys
import os
import pandas as pd
import time

# Ensure Python finds your engine files
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

from semantic_join import semantic_join
from utils import ground_truth

def run_sanity_check():
    # 1. Load Data & Ground Truth
    print("Loading data...")
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')
    
    predicate_str = "both reviews express the same sentiment (both positive or both negative)"
    predicate_fn = lambda a, b: a["sentiment"] == b["sentiment"]
    gt_pairs = ground_truth(table_a, table_b, predicate_fn)
    
    print(f"Total True Positive Pairs (Ground Truth): {len(gt_pairs)}\n")
    print("Executing Sanity Check with GPT-4o...")
    
    # 2. Run Pipeline (Using configuration from Run 15, but with gpt-4o)
    try:
        t0 = time.time()
        result = semantic_join(
            table_a, table_b,
            predicate=predicate_str,
            schema_a=["review"],
            schema_b=["review"],
            llm_model="gpt-4o",                # THE CRITICAL CHANGE
            clustering="kmeans",
            embedding="all-mpnet-base-v2",     # From Run 15
            n_clusters=2,                      # From Run 15
            filter_threshold=0.15,             # From Run 15
            block_size=10,                     # From Run 15
            cluster_size_limit=150,            
            verbose=True                       # Keep true to watch extraction progress
        )
        total_time = time.time() - t0
        
        # 3. Compute Metrics
        pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
        tp = len(gt_pairs & pred)
        fp = len(pred - gt_pairs)
        fn = len(gt_pairs - pred)
        
        recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
        precision = (tp / len(pred) * 100) if pred else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        
        # 4. Output Results
        print("\n" + "="*50)
        print("SANITY CHECK RESULTS: gpt-4o")
        print("="*50)
        print(f"True Positives Found : {tp} / {len(gt_pairs)}")
        print(f"False Positives      : {fp}")
        print(f"False Negatives      : {fn}")
        print("-" * 50)
        print(f"Recall               : {recall:.2f}%  <-- THIS IS THE METRIC TO WATCH")
        print(f"Precision            : {precision:.2f}%")
        print(f"F1 Score             : {f1:.2f}%")
        print("-" * 50)
        print(f"Total Tokens Used    : {result.tokens.total:,}")
        print(f"Total Execution Time : {total_time:.1f} seconds")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during sanity check: {e}")

if __name__ == "__main__":
    run_sanity_check()