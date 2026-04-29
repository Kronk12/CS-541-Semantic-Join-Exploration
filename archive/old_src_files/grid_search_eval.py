import sys
import os
import itertools
import time
import pandas as pd
import logging

# Ensure Python finds your engine files
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

from semantic_join import semantic_join
from utils import ground_truth

def get_average_purity(df, labels):
    """Calculates the weighted average purity of clusters."""
    if labels is None or len(labels) == 0:
        return 0.0
    
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    total_rows = 0
    total_majority_class = 0
    
    for c in sorted(df_clustered['cluster'].unique()):
        if c < 0: continue # Skip noise
        subset = df_clustered[df_clustered['cluster'] == c]
        if len(subset) == 0: continue
        
        counts = subset['sentiment'].value_counts()
        total_rows += len(subset)
        total_majority_class += counts.max()
        
    return (total_majority_class / total_rows) * 100 if total_rows > 0 else 0.0

def run_experiments():
    # 1. Load Data & Ground Truth
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')
    
    predicate_str = "both reviews express the same sentiment (both positive or both negative)"
    predicate_fn = lambda a, b: a["sentiment"] == b["sentiment"]
    gt_pairs = ground_truth(table_a, table_b, predicate_fn)
    
    # 2. Define Hyperparameter Grid
    param_grid = {
        "embedding": [
            "distilbert-base-uncased-finetuned-sst-2-english", 
            "all-mpnet-base-v2"
        ],
        "n_clusters": [2, 5],
        "filter_threshold": [0.0, 0.15, 0.25],
        "block_size": [10, 20]
    }
    
    # Generate all combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print(f"Starting Grid Search: {len(experiments)} configurations to test.")
    
    # 3. Execute Grid Search
    for i, config in enumerate(experiments, 1):
        print(f"\n--- Running Experiment {i}/{len(experiments)} ---")
        print(f"Config: {config}")
        
        try:
            # Force pairwise to test clustering parameters
            result = semantic_join(
                table_a, table_b,
                predicate=predicate_str,
                schema_a=["review"],
                schema_b=["review"],
                llm_model="gpt-4o-mini", # Use mini to save costs during grid search
                clustering="kmeans",
                embedding=config["embedding"],
                n_clusters=config["n_clusters"],
                filter_threshold=config["filter_threshold"],
                block_size=config["block_size"],
                cluster_size_limit=config["block_size"]**2 + 50, # Scale limit with block size
                verbose=False
            )
            
            # Compute Final Eval Metrics
            pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
            tp = len(gt_pairs & pred)
            fp = len(pred - gt_pairs)
            fn = len(gt_pairs - pred)
            recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
            precision = (tp / len(pred) * 100) if pred else 0.0
            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
            
            # Compute Purity
            purity_a = get_average_purity(table_a, result.labels_a)
            purity_b = get_average_purity(table_b, result.labels_b)
            avg_purity = (purity_a + purity_b) / 2
            
            # Compute Filtering Loss
            initial_pairs = set(result.cluster_pairs_initial or [])
            surviving_pairs = set(result.cluster_pairs_surviving or [])
            dropped_cluster_pairs = initial_pairs - surviving_pairs
            
            dropped_row_pairs = set()
            for (ca, cb) in dropped_cluster_pairs:
                indices_a = [idx for idx, lbl in enumerate(result.labels_a) if lbl == ca]
                indices_b = [idx for idx, lbl in enumerate(result.labels_b) if lbl == cb]
                for a_idx in indices_a:
                    for b_idx in indices_b:
                        dropped_row_pairs.add((a_idx, b_idx))
                        
            tps_lost = dropped_row_pairs.intersection(gt_pairs)
            percent_tps_lost = (len(tps_lost) / len(gt_pairs)) * 100 if gt_pairs else 0.0
            
            # Log Data
            run_data = {
                "Run_ID": i,
                "Embedding": config["embedding"],
                "K_Clusters": config["n_clusters"],
                "Filter_Threshold": config["filter_threshold"],
                "Block_Size": config["block_size"],
                "Avg_Purity_%": round(avg_purity, 2),
                "Dropped_Clusters": len(dropped_cluster_pairs),
                "TPs_Lost_to_Filter": len(tps_lost),
                "Percent_TPs_Lost": round(percent_tps_lost, 2),
                "Recall": round(recall, 2),
                "Precision": round(precision, 2),
                "F1": round(f1, 2),
                "Total_Tokens": result.tokens.total,
                "Total_Time_s": round(sum(result.timings.values()), 2)
            }
            results.append(run_data)
            
            # Save incrementally in case of a crash
            pd.DataFrame(results).to_csv("hyperparameter_results.csv", index=False)
            print(f"Success! Recall: {recall:.1f}%, Tokens: {result.tokens.total:,}, Purity: {avg_purity:.1f}%")
            
        except Exception as e:
            print(f"Error on config {config}: {e}")

    print("\nGrid Search Complete! Results saved to 'hyperparameter_results.csv'.")

if __name__ == "__main__":
    run_experiments()