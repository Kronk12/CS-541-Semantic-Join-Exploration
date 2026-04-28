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

@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RateLimitError), reraise=True)
def safe_semantic_join(*args, **kwargs):
    return semantic_join(*args, **kwargs)

def run_emails_pairwise():
    print("Loading Emails Dataset...")
    # Sampling Table A to 100 to match the scale of the other tests
    emails_a = pd.read_csv('data/table_a_emails.csv').sample(100, random_state=42).reset_index(drop=True)
    emails_b = pd.read_csv('data/table_b_emails.csv')
    
    gt_pairs = ground_truth(
        emails_a, emails_b, 
        lambda a, b: (a["name"] == b["name"]) and (b["month_idx"] < a["month_idx"])
    )
    
    # Ratios for dynamic K calculation inside the engine
    ratios = [10, 20] 
    # Bumped thresholds to force the filter to drop unpromising clusters
    thresholds = [0.10, 0.20, 0.30]
    trials = 1
    
    results = []
    output_file = os.path.join(current_dir, 'logs/emails_pairwise_showdown.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nStarting Emails Pairwise Showdown. Saving to {output_file}\n" + "="*50)
    
    run_counter = 1
    for ratio in ratios:
        for thresh in thresholds:
            for trial in range(1, trials + 1):
                print(f"--- Run {run_counter} | Emails | Ratio 1:{ratio} | Thresh={thresh} | Trial {trial}/{trials} ---")
                
                try:
                    result = safe_semantic_join(
                        emails_a, emails_b,
                        predicate="the two texts contradict each other",
                        schema_a=["review"], schema_b=["review"],
                        llm_model="gpt-4o", 
                        
                        force_strategy="pairwise",
                        force_projection=False,  # Raw text cross-row logic
                        embedding="all-mpnet-base-v2",
                        clustering="kmeans",
                        
                        # NEW ENGINE PARAMETER
                        cluster_ratio=ratio,
                        n_clusters=None,
                        filter_threshold=thresh,
                        
                        block_size=15,            
                        cluster_size_limit=500,
                        min_profile_size=2,       
                        filter_sample_size=5,
                        verbose=False
                    )
                    
                    pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
                    tp = len(gt_pairs & pred)
                    recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
                    precision = (tp / len(pred) * 100) if pred else 0.0
                    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
                    
                    initial_pairs = set(result.cluster_pairs_initial or [])
                    surviving_pairs = set(result.cluster_pairs_surviving or [])
                    dropped = len(initial_pairs) - len(surviving_pairs)
                    
                    results.append({
                        "Run_ID": run_counter, "Dataset": "Emails", "Strategy": "pairwise",
                        "Trial": trial, "Row_Ratio": f"1:{ratio}", "Threshold": thresh, "Dropped_Clusters": dropped,
                        "Recall": round(recall, 2), "Precision": round(precision, 2), "F1": round(f1, 2),
                        "Tokens": result.tokens.total, "Time_s": round(sum(result.timings.values()), 2)
                    })
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    
                    print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,} | Dropped: {dropped}")
                    run_counter += 1
                    time.sleep(2)
                except Exception as e:
                    print(f"  -> Error: {e}")
                    time.sleep(5)

if __name__ == "__main__":
    run_emails_pairwise()