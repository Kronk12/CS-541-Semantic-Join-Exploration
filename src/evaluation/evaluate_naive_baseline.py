import sys
import os
import pandas as pd
import time
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

from cluster_join import join_cluster_pair
from utils import ground_truth

@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RateLimitError), reraise=True)
def safe_naive_join(*args, **kwargs):
    return join_cluster_pair(*args, **kwargs)

def load_extrapolation_datasets():
    print("Loading datasets for Naive Extrapolation...")

    so_gt_df = pd.read_csv('data/stack_ground_truth.csv')
    so_gt_pairs = set(zip(so_gt_df['question_id'], so_gt_df['concept_id']))
    
    return {
        "IMDB": {
            "a": pd.read_csv('data/table_a.csv'),
            "b": pd.read_csv('data/table_b.csv'),
            "schema_a": ["review"], "schema_b": ["review"],
            "pred": "Both reviews express the same sentiment (Positive or Negative)",
            "gt_fn": lambda a, b: a["sentiment"] == b["sentiment"],
            "sample_a": 10, "sample_b": 10
        },
        "Emails": {
            "a": pd.read_csv('data/table_a_emails.csv'),
            "b": pd.read_csv('data/table_b_emails.csv'),
            "schema_a": ["email"], "schema_b": ["email"],
            "pred": "the two texts contradict each other",
            "gt_fn": lambda a, b: (a["name"] == b["name"]) and (b["month_idx"] < a["month_idx"]),
            "sample_a": 10, "sample_b": 10
        },
        "StackOverflow": {
            "a": pd.read_csv('data/table_a_stack.csv'),
            "b": pd.read_csv('data/table_b_stack.csv'),
            "schema_a": ["question_text"], 
            "schema_b": ["concept_name", "Description"],
            "pred": "The question describes symptoms, errors, or intents that are solved by or directly related to this programming concept.",
            "gt_fn": lambda a, b: (a["question_id"], b["concept_id"]) in so_gt_pairs,
            "sample_a": 10, "sample_b": 10 
        }
    }

def run_naive_extrapolated_baseline():
    datasets = load_extrapolation_datasets()
    trials = 3
    results = []
    output_file = os.path.join(current_dir, 'logs/baseline_naive_extrapolated_report.csv')
    
    print(f"\nStarting NAIVE Baseline (Block=1) with Extrapolation. Saving to {output_file}")
    
    run_counter = 1
    for ds_name, ds in datasets.items():
        print(f"\n{'='*50}\nDATASET: {ds_name}\n{'='*50}")
        
        full_a_len = len(ds["a"])
        full_b_len = len(ds["b"])
        total_pairs_in_full_dataset = full_a_len * full_b_len
        
        for trial in range(1, trials + 1):
            sampled_a = ds["a"].sample(n=ds["sample_a"], random_state=trial*10)
            sampled_b = ds["b"].sample(n=ds["sample_b"], random_state=trial*100) if ds["sample_b"] else ds["b"]
            
            gt_pairs = ground_truth(sampled_a, sampled_b, ds["gt_fn"])
            pairs_in_sample = len(sampled_a) * len(sampled_b)
            
            multiplier = total_pairs_in_full_dataset / pairs_in_sample
            
            print(f"--- Run {run_counter} | Trial {trial}/{trials} | Sample: {pairs_in_sample} pairs | Projecting to: {total_pairs_in_full_dataset} pairs ---")
            
            try:
                t0 = time.time()
                
                result = safe_naive_join(
                    cluster_a=sampled_a, cluster_b=sampled_b,
                    predicate=ds["pred"],
                    schema_a=ds["schema_a"], schema_b=ds["schema_b"],
                    llm_model="gpt-4o", 
                    block_size=1,                
                    cluster_size_limit=-1,       
                    max_chars=400,
                    verbose=False
                )
                
                actual_time = time.time() - t0
                
                pred = set(result.matches)
                tp = len(gt_pairs & pred)
                recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
                precision = (tp / len(pred) * 100) if pred else 0.0
                f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
                
                actual_tokens = result.tokens.total
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