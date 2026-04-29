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

def run_mts_pairwise():
    print("Loading MTSamples...")
    # Sample Table A down to 100 to match your IMDB/Emails scale for fair token/time comparison
    mts_a = pd.read_csv('data/table_a_transcripts.csv').sample(100, random_state=42).reset_index(drop=True)
    mts_b = pd.read_csv('data/table_b_specialties.csv')
    
    gt_pairs = ground_truth(
        mts_a, mts_b, 
        lambda a, b: a["medical_specialty"] == b["specialty"]
    )
    
    # We will test two cluster ratios and two thresholds
    ratios = [10, 20] 
    thresholds = [0.10, 0.15, 0.20]
    trials = 1
    
    results = []
    output_file = os.path.join(current_dir, 'logs/mts_pairwise_showdown.csv')
    print(f"\nStarting MTSamples Pairwise Showdown. Saving to {output_file}\n" + "="*50)
    
    run_counter = 1
    for ratio in ratios:
        k = max(2, len(mts_a) // ratio)
        for thresh in thresholds:
            for trial in range(1, trials + 1):
                print(f"--- Run {run_counter} | Pairwise + Projection | Ratio 1:{ratio} (K={k}) | Thresh={thresh} | Trial {trial}/{trials} ---")
                
                try:
                    result = safe_semantic_join(
                        mts_a, mts_b,
                        predicate="The clinical transcription belongs to this medical specialty.",
                        schema_a=["transcription"], schema_b=["specialty"],
                        llm_model="gpt-4o", 
                        
                        # THE CRITICAL SETTINGS FOR THIS TEST
                        force_strategy="pairwise",
                        force_projection=True,         # Must project transcripts to specialties
                        embedding="all-mpnet-base-v2",
                        clustering="kmeans",
                        n_clusters=k,
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
                        "Run_ID": run_counter, "Dataset": "MTSamples", "Strategy": "pairwise (Projected)",
                        "Trial": trial, "K_Clusters": k, "Threshold": thresh, "Dropped_Clusters": dropped,
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
    run_mts_pairwise()