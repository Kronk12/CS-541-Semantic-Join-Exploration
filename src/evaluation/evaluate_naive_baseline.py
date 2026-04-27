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

def load_extrapolation_datasets():
    print("Loading datasets for Naive Extrapolation...")
    df_imdb = pd.read_csv('data/IMDB dataset.csv')
    
    return {
        "IMDB": {
            "a": df_imdb.sample(n=100, random_state=42).reset_index(drop=True),
            "b": df_imdb.sample(n=100, random_state=100).reset_index(drop=True),
            "schema_a": ["review"], "schema_b": ["review"],
            "pred": "both reviews express the same sentiment",
            "gt_fn": lambda a, b: a["sentiment"] == b["sentiment"],
            "sample_a": 10, "sample_b": 10  # 100 pairs evaluated
        },
        "Emails": {
            "a": pd.read_csv('data/table_a_emails.csv'),
            "b": pd.read_csv('data/table_b_emails.csv'),
            "schema_a": ["review"], "schema_b": ["review"],
            "pred": "the two texts contradict each other",
            "gt_fn": lambda a, b: (a["name"] == b["name"]) and (b["month_idx"] < a["month_idx"]),
            "sample_a": 10, "sample_b": 10  # 100 pairs evaluated
        },
        "MTSamples": {
            "a": pd.read_csv('data/table_a_transcripts.csv'),
            "b": pd.read_csv('data/table_b_specialties.csv'),
            "schema_a": ["transcription"], "schema_b": ["specialty"],
            "pred": "The clinical transcription belongs to this medical specialty.",
            "gt_fn": lambda a, b: a["medical_specialty"] == b["specialty"],
            "sample_a": 5, "sample_b": None  # 5x40 = 200 pairs evaluated
        }
    }

def run_naive_extrapolated_baseline():
    datasets = load_extrapolation_datasets()
    trials = 3
    results = []
    output_file = os.path.join(current_dir, 'baseline_naive_extrapolated_report.csv')
    
    print(f"\nStarting NAIVE Baseline (Block=1) with Extrapolation. Saving to {output_file}")
    
    run_counter = 1
    for ds_name, ds in datasets.items():
        print(f"\n{'='*50}\nDATASET: {ds_name}\n{'='*50}")
        
        # Calculate full dataset footprint
        full_a_len = len(ds["a"])
        full_b_len = len(ds["b"])
        total_pairs_in_full_dataset = full_a_len * full_b_len
        
        for trial in range(1, trials + 1):
            sampled_a = ds["a"].sample(n=ds["sample_a"], random_state=trial*10)
            sampled_b = ds["b"].sample(n=ds["sample_b"], random_state=trial*100) if ds["sample_b"] else ds["b"]
            
            # Ground truth just for the sample to get accurate F1 metrics
            gt_pairs = ground_truth(sampled_a, sampled_b, ds["gt_fn"])
            pairs_in_sample = len(sampled_a) * len(sampled_b)
            
            # The Extrapolation Factor
            multiplier = total_pairs_in_full_dataset / pairs_in_sample
            
            print(f"--- Run {run_counter} | Trial {trial}/{trials} | Sample: {pairs_in_sample} pairs | Projecting to: {total_pairs_in_full_dataset} pairs ---")
            
            try:
                result = safe_semantic_join(
                    sampled_a, sampled_b,
                    predicate=ds["pred"],
                    schema_a=ds["schema_a"], schema_b=ds["schema_b"],
                    llm_model="gpt-4o", 
                    
                    force_strategy="pairwise",
                    force_projection=False,      
                    n_clusters=2,                
                    filter_threshold=0.0,        
                    block_size=1,                # <--- THE NAIVE SETTING
                    verbose=False
                )
                
                # Sample Metrics
                pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
                tp = len(gt_pairs & pred)
                recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
                precision = (tp / len(pred) * 100) if pred else 0.0
                f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
                
                # Extrapolate Costs
                actual_tokens = result.tokens.total
                actual_time = sum(result.timings.values())
                projected_tokens = int(actual_tokens * multiplier)
                projected_time = actual_time * multiplier
                
                results.append({
                    "Run_ID": run_counter, "Dataset": ds_name, "Trial": trial,
                    "Baseline_Type": "Naive_Block_1_Extrapolated",
                    "Extrapolation_Multiplier": round(multiplier, 2),
                    "Recall": round(recall, 2), "Precision": round(precision, 2), "F1": round(f1, 2),
                    "Actual_Tokens": actual_tokens,
                    "Projected_Full_Tokens": projected_tokens,
                    "Actual_Time_s": round(actual_time, 2),
                    "Projected_Full_Time_s": round(projected_time, 2)
                })
                pd.DataFrame(results).to_csv(output_file, index=False)
                
                print(f"  -> R:{recall:.1f}% P:{precision:.1f}%")
                print(f"  -> Actual Tokens: {actual_tokens:,} | Projected Tokens: {projected_tokens:,}")
                
                run_counter += 1
                time.sleep(2)
            except Exception as e:
                print(f"  -> Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    run_naive_extrapolated_baseline()