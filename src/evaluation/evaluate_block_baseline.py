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
def safe_block_join(*args, **kwargs):
    return join_cluster_pair(*args, **kwargs)

def load_full_datasets():
    print("Loading datasets for FULL Block evaluation...")
    so_gt_df = pd.read_csv('data/stack_ground_truth.csv')
    so_gt_pairs = set(zip(so_gt_df['question_id'], so_gt_df['concept_id']))
    
    return {
        "IMDB": {
            "a": pd.read_csv('data/table_a_50.csv'),
            "b": pd.read_csv('data/table_b_50.csv'),
            "schema_a": ["review"], "schema_b": ["review"],
            "pred": "Both reviews express the same sentiment (Positive or Negative)",
            "gt_fn": lambda a, b: a["sentiment"] == b["sentiment"],
        },
        "Emails": {
            "a": pd.read_csv('data/table_a_emails.csv'),
            "b": pd.read_csv('data/table_b_emails.csv'),
            "schema_a": ["email"], "schema_b": ["email"],
            "pred": "the two texts contradict each other",
            "gt_fn": lambda a, b: (a["name"] == b["name"]) and (b["month_idx"] < a["month_idx"]),
        },
        "StackOverflow": {
            "a": pd.read_csv('data/table_a_stack.csv'),
            "b": pd.read_csv('data/table_b_stack.csv'),
            "schema_a": ["question_text"], 
            "schema_b": ["concept_name", "Description"],
            "pred": "The question describes symptoms, errors, or intents that are solved by or directly related to this programming concept.",
            "gt_fn": lambda a, b: (a["question_id"], b["concept_id"]) in so_gt_pairs,
        }
    }

def run_full_block_baseline():
    datasets = load_full_datasets()
    block_sizes = [5, 10, 15, 20, 25]
    trials = 2
    results = []
    output_file = os.path.join(current_dir, 'logs/baseline_block_size_tests_continued.csv')
    
    print(f"\nStarting FULL BLOCK Baseline. Saving to {output_file}")
    
    run_counter = 1
    for ds_name, ds in datasets.items():
        print(f"\n{'='*50}\nDATASET: {ds_name}\n{'='*50}")
        
        gt_pairs = ground_truth(ds["a"], ds["b"], ds["gt_fn"])
        total_pairs = len(ds["a"]) * len(ds["b"])
        
        for bs in block_sizes:
            for trial in range(1, trials + 1):
                print(f"--- Run {run_counter} | Block: {bs} | Trial {trial}/{trials} | Grid: {len(ds['a'])}x{len(ds['b'])} ({total_pairs} pairs) ---")
                
                try:
                    t0 = time.time()
                    
                    result = safe_block_join(
                        cluster_a=ds["a"], cluster_b=ds["b"],
                        predicate=ds["pred"],
                        schema_a=ds["schema_a"], schema_b=ds["schema_b"],
                        llm_model="gpt-4o", 
                        block_size=bs,               
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
                    
                    results.append({
                        "Run_ID": run_counter, "Dataset": ds_name, "Trial": trial,
                        "Baseline_Type": f"Block_{bs}_Full",
                        "A_Size": len(ds["a"]), "B_Size": len(ds["b"]),
                        "Total_Pairs_Evaluated": total_pairs,
                        "Recall": round(recall, 2), "Precision": round(precision, 2), "F1": round(f1, 2),
                        "Tokens": result.tokens.total, "Time_s": round(actual_time, 2)
                    })
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    
                    print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,}")
                    run_counter += 1
                    time.sleep(2)
                except Exception as e:
                    print(f"  -> Error: {e}")
                    time.sleep(5)

if __name__ == "__main__":
    run_full_block_baseline()