import sys
import os
import itertools
import pandas as pd
import time

# Ensure Python finds your engine files
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

from semantic_join import semantic_join
from utils import ground_truth

def get_average_purity(df, labels):
    if labels is None or len(labels) == 0:
        return 0.0
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    total_rows, total_majority_class = 0, 0
    for c in sorted(df_clustered['cluster'].unique()):
        if c < 0: continue
        subset = df_clustered[df_clustered['cluster'] == c]
        if len(subset) == 0: continue
        total_rows += len(subset)
        total_majority_class += subset['sentiment'].value_counts().max()
    return (total_majority_class / total_rows) * 100 if total_rows > 0 else 0.0

def run_block_grid_search():
    print("Loading data...")
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')
    
    predicate_str = "both reviews express the same sentiment (both positive or both negative)"
    predicate_fn = lambda a, b: a["sentiment"] == b["sentiment"]
    gt_pairs = ground_truth(table_a, table_b, predicate_fn)
    
    # --- HYPERPARAMETER GRID ---
    param_grid = {
        "embedding": ["distilbert-base-uncased-finetuned-sst-2-english", "all-mpnet-base-v2"],
        "block_size": [5, 10, 15],
        "filter_threshold": [0.0, 0.15, 0.30]
    }
    
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []
    
    # Target output file
    output_file = os.path.join(current_dir, 'grid_search_blocks.csv')
    print(f"Starting Grid Search: {len(experiments)} configs. Saving to {output_file}")
    
    for i, config in enumerate(experiments, 1):
        print(f"\n--- Run {i}/{len(experiments)} --- | {config}")
        try:
            # Force pairwise, run with GPT-4o for high extraction recall
            result = semantic_join(
                table_a, table_b,
                predicate=predicate_str,
                schema_a=["review"], schema_b=["review"],
                llm_model="gpt-4o", 
                clustering="kmeans",
                n_clusters=2,
                embedding=config["embedding"],
                filter_threshold=config["filter_threshold"],
                block_size=config["block_size"],
                cluster_size_limit=config["block_size"]**2 + 50,
                min_profile_size=10,  # CRITICAL: Forces the filter to run on small datasets
                filter_sample_size=5, # Take 5 from each cluster for the mini-join
                verbose=False
            )
            
            # Metrics
            pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
            tp = len(gt_pairs & pred)
            fp = len(pred - gt_pairs)
            fn = len(gt_pairs - pred)
            recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
            precision = (tp / len(pred) * 100) if pred else 0.0
            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
            
            avg_purity = (get_average_purity(table_a, result.labels_a) + get_average_purity(table_b, result.labels_b)) / 2
            
            initial_pairs = set(result.cluster_pairs_initial or [])
            surviving_pairs = set(result.cluster_pairs_surviving or [])
            dropped_cluster_pairs = len(initial_pairs) - len(surviving_pairs)
            
            run_data = {
                "Run_ID": i,
                "Embedding": config["embedding"],
                "Block_Size": config["block_size"],
                "Filter_Threshold": config["filter_threshold"],
                "Avg_Purity_%": round(avg_purity, 2),
                "Dropped_Clusters": dropped_cluster_pairs,
                "Recall": round(recall, 2),
                "Precision": round(precision, 2),
                "F1": round(f1, 2),
                "Tokens": result.tokens.total,
                "Time_s": round(sum(result.timings.values()), 2)
            }
            results.append(run_data)
            
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,} | Dropped Clusters:{dropped_cluster_pairs}")
            
        except Exception as e:
            print(f"  -> Error: {e}")

if __name__ == "__main__":
    run_block_grid_search()