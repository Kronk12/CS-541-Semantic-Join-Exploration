import sys
import os
import pandas as pd
import time
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

from semantic_join import semantic_join
from utils import ground_truth

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60), 
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def safe_semantic_join(*args, **kwargs):
    return semantic_join(*args, **kwargs)

def load_datasets():
    print("Loading all datasets...")
    # 1. IMDB 
    df_imdb = pd.read_csv('data/IMDB dataset.csv')
    imdb_a = df_imdb.sample(n=100, random_state=42).reset_index(drop=True)
    imdb_b = df_imdb.sample(n=100, random_state=100).reset_index(drop=True)
    
    # 2. Emails
    emails_a = pd.read_csv('data/table_a_emails.csv')
    emails_b = pd.read_csv('data/table_b_emails.csv')
    
    # 3. MTSamples
    mts_a = pd.read_csv('data/table_a_transcripts.csv')
    mts_b = pd.read_csv('data/table_b_specialties.csv')
    
    return [
        # {
        #     "dataset_name": "IMDB",
        #     "table_a": imdb_a, "table_b": imdb_b,
        #     "schema_a": ["review"], "schema_b": ["review"],
        #     "predicate": "both reviews express the same sentiment",
        #     "ground_truth_fn": lambda a, b: a["sentiment"] == b["sentiment"],
        #     "optimal_emb": None, 
        #     "optimal_proj": False,
        #     "dynamic_labels": list(imdb_b['sentiment'].unique()), 
        #     "strategies_to_test": ["classifier"] 
        # },
        
        {
            "dataset_name": "MTSamples",
            "table_a": mts_a, "table_b": mts_b,
            "schema_a": ["transcription"], "schema_b": ["specialty"],
            "predicate": "The clinical transcription belongs to this medical specialty.",
            "ground_truth_fn": lambda a, b: a["medical_specialty"] == b["specialty"],
            "optimal_emb": "all-mpnet-base-v2", 
            "optimal_proj": True,
            "dynamic_labels": list(mts_b['specialty'].unique()), 
            "strategies_to_test": ["classifier", "pairwise"] 
        },
        {
            "dataset_name": "Emails",
            "table_a": emails_a, "table_b": emails_b,
            "schema_a": ["review"], "schema_b": ["review"],
            "predicate": "the two texts contradict each other",
            "ground_truth_fn": lambda a, b: (a["name"] == b["name"]) and (b["month_idx"] < a["month_idx"]),
            "optimal_emb": "all-mpnet-base-v2",
            "optimal_proj": False,
            "dynamic_labels": None,
            "strategies_to_test": ["pairwise"] 
        }
    ]

def run_comprehensive_grid_search():
    datasets = load_datasets()
    
    # Grid parameters
    # ratio_grid = [10, 20] 
    # threshold_grid = [0.0, 0.025, 0.05, 0.075, 0.10]
    ratio_grid = [10]
    threshold_grid = [0.0, 0.10, 0.20]
    trials_per_config = 3
    
    results = []
    output_file = os.path.join(current_dir, 'grid_search_final_report_2.csv')
    print(f"\nStarting Master Grid Search. Saving to {output_file}")
    
    run_counter = 1
    for ds in datasets:
        print(f"\n{'='*50}\nDATASET: {ds['dataset_name']}\n{'='*50}")
        gt_pairs = ground_truth(ds['table_a'], ds['table_b'], ds['ground_truth_fn'])
        n_rows = len(ds['table_a'])
        
        for strategy in ds["strategies_to_test"]:
            is_classification = (strategy == "classifier")
            
            # Setup grid loops
            current_ratios = [None] if is_classification else ratio_grid
            current_thresh_grid = [0.0] if is_classification else threshold_grid

            for ratio in current_ratios:
                k = max(2, n_rows // ratio) if ratio else None
                
                for thresh in current_thresh_grid:
                    mode_label = f"Strategy: {strategy}" + (f" | Ratio=1:{ratio} (K={k}) | T={thresh}" if not is_classification else "")
                    
                    for trial in range(1, trials_per_config + 1):
                        print(f"--- Run {run_counter} | {mode_label} | Trial {trial}/{trials_per_config} ---")
                        
                        try:
                            result = safe_semantic_join(
                                ds['table_a'], ds['table_b'],
                                predicate=ds['predicate'],
                                schema_a=ds['schema_a'], schema_b=ds['schema_b'],
                                llm_model="gpt-4o", 
                                
                                # Apply Overrides
                                force_strategy=strategy,
                                force_projection=ds['optimal_proj'] if not is_classification else False,
                                force_labels=ds['dynamic_labels'] if is_classification else None,
                                
                                # Pairwise specific params
                                embedding=ds['optimal_emb'] if not is_classification else None,
                                clustering="kmeans" if not is_classification else None,
                                n_clusters=k,
                                filter_threshold=thresh,
                                block_size=15,            
                                cluster_size_limit=500,
                                min_profile_size=2,       
                                filter_sample_size=5,
                                verbose=False
                            )
                            
                            # Metrics
                            pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
                            tp = len(gt_pairs & pred)
                            recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
                            precision = (tp / len(pred) * 100) if pred else 0.0
                            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
                            
                            # Calculate dropped clusters 
                            initial_pairs = set(result.cluster_pairs_initial or [])
                            surviving_pairs = set(result.cluster_pairs_surviving or [])
                            dropped = len(initial_pairs) - len(surviving_pairs) if not is_classification else "N/A"
                            
                            run_data = {
                                "Run_ID": run_counter,
                                "Dataset": ds["dataset_name"],
                                "Strategy": strategy,
                                "Trial_ID": trial,
                                "Row_Ratio": f"1:{ratio}" if ratio else "N/A",
                                "K_Clusters": k if k else "N/A",
                                "Threshold": thresh if not is_classification else "N/A",
                                "Dropped_Clusters": dropped,
                                "Recall": round(recall, 2),
                                "Precision": round(precision, 2),
                                "F1": round(f1, 2),
                                "Tokens": result.tokens.total,
                                "Time_s": round(sum(result.timings.values()), 2)
                            }
                            results.append(run_data)
                            pd.DataFrame(results).to_csv(output_file, index=False)
                            
                            print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,}")
                            run_counter += 1
                            time.sleep(2)
                            
                        except Exception as e:
                            print(f"  -> Error: {e}")
                            time.sleep(5)

if __name__ == "__main__":
    run_comprehensive_grid_search()