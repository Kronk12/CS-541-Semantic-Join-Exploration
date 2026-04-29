import sys
import os
import itertools
import pandas as pd
import time
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

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

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60), 
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def safe_semantic_join(*args, **kwargs):
    """Wraps the pipeline execution to catch OpenAI 429 errors and automatically wait/retry."""
    return semantic_join(*args, **kwargs)

def run_imdb_scaling_search():
    print("Loading IMDB dataset...")
    imdb_path = os.path.join(current_dir, '..', '..', 'data', 'IMDB dataset.csv')
    df_full = pd.read_csv(imdb_path)
    
    predicate_str = "both reviews express the same sentiment (both positive or both negative)"
    predicate_fn = lambda a, b: a["sentiment"] == b["sentiment"]
    
    # --- SCALING HYPERPARAMETER GRID ---
    param_grid = {
        "table_size": [100, 250, 500],             
        "cluster_ratio": [0.05, 0.10, .15],       # Calculates K dynamically (e.g., 5% of 500 = 25 clusters)
        "filter_threshold": [0.35]           
    }
    
    trials_per_config = 1
    locked_block_size = 30
    locked_embedding = "distilbert-base-uncased-finetuned-sst-2-english"
    
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []
    
    output_file = os.path.join(current_dir, 'grid_search_imdb_scale.csv')
    print(f"Starting IMDB Scaling Search: {len(experiments)} configs x {trials_per_config} trials = {len(experiments)*trials_per_config} total runs.")
    print(f"Saving to {output_file}\n")
    
    run_counter = 1
    for i, config in enumerate(experiments, 1):
        for trial in range(1, trials_per_config + 1):
            # Calculate actual K for this run (ensure at least 2 clusters)
            calculated_k = max(2, int(config["table_size"] * config["cluster_ratio"]))
            
            print(f"--- Config {i}/{len(experiments)} | Trial {trial}/{trials_per_config} ---")
            print(f"Parameters: Table Size={config['table_size']}, K_Clusters={calculated_k} (Ratio: {config['cluster_ratio']}), Thresh={config['filter_threshold']}")
            
            try:
                # 1. Take random samples and isolate them 
                table_a = df_full.sample(n=config["table_size"], random_state=42 + trial).reset_index(drop=True)
                table_b = df_full.sample(n=config["table_size"], random_state=100 + trial).reset_index(drop=True)
                
                # 2. Calculate ground truth for this specific trial's data
                gt_pairs = ground_truth(table_a, table_b, predicate_fn)
                
                # 3. Execute Pipeline
                result = safe_semantic_join(
                    table_a, table_b,
                    predicate=predicate_str,
                    schema_a=["review"], schema_b=["review"],
                    llm_model="gpt-4o", 
                    clustering="kmeans",
                    n_clusters=calculated_k,          # Inject dynamic K here
                    embedding=locked_embedding,
                    filter_threshold=config["filter_threshold"],
                    block_size=locked_block_size,
                    cluster_size_limit=locked_block_size**2 + 50,
                    min_profile_size=10,  
                    filter_sample_size=5, 
                    verbose=False
                )
                
                # 4. Calculate Metrics
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
                
                # 5. Log Data
                run_data = {
                    "Run_ID": run_counter,
                    "Config_ID": i,
                    "Trial_ID": trial,
                    "Table_Size": config["table_size"],
                    "Cluster_Ratio": config["cluster_ratio"],
                    "Calculated_K": calculated_k,
                    "Filter_Threshold": config["filter_threshold"],
                    "Avg_Purity_%": round(avg_purity, 2),
                    "Total_Clusters_Possible": calculated_k**2,
                    "Dropped_Clusters": dropped_cluster_pairs,
                    "Recall": round(recall, 2),
                    "Precision": round(precision, 2),
                    "F1": round(f1, 2),
                    "Tokens": result.tokens.total,
                    "Time_s": round(sum(result.timings.values()), 2)
                }
                results.append(run_data)
                
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,} | Dropped: {dropped_cluster_pairs}/{calculated_k**2}")
                
                run_counter += 1
                time.sleep(2) # Brief cooldown for OpenAI
                
            except Exception as e:
                print(f"  -> Permanent Error on Config {i}, Trial {trial}: {e}")
                time.sleep(5)

if __name__ == "__main__":
    run_imdb_scaling_search()