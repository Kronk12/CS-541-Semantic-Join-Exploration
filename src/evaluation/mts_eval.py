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
    """Wraps the pipeline execution to catch OpenAI 429 errors and automatically wait/retry."""
    return semantic_join(*args, **kwargs)

def run_medical_search():
    print("Loading MTSamples dataset...")
    # Make sure your prepared CSVs are in the data/ directory
    table_a = pd.read_csv('data/table_a_transcripts.csv')
    table_b = pd.read_csv('data/table_b_specialties.csv')
    
    predicate_str = "The clinical transcription belongs to this medical specialty."
    
    # Ground truth logic: Match the doctor's original label to the taxonomy
    predicate_fn = lambda a, b: a["medical_specialty"] == b["specialty"]
    
    gt_pairs = ground_truth(table_a, table_b, predicate_fn)
    print(f"Ground Truth Matches: {len(gt_pairs)}")
    
    emb = "all-mpnet-base-v2"
    
    # 5 Explicit Configurations to test the baselines, the filter, and the advisor
    configs = [
        # {"proj": False, "thresh": 0.0,  "label": "Baseline_Standard"},
        {"proj": True,  "thresh": 0.05,  "label": "0.05_projected"},
        # {"proj": False, "thresh": 0.15, "label": "Filter_Standard"},
        {"proj": True,  "thresh": 0.1, "label": "0.1_projected"},
        {"proj": True,  "thresh": 0.15, "label": "0.15_projected"}
        # {"proj": None,  "thresh": 0.15, "label": "Filter_Auto_Advisor"}
    ]
    
    results = []
    output_file = os.path.join(current_dir, 'grid_search_mtsamples_2.csv')
    
    print(f"Starting Medical Specialty Evaluation. Saving to {output_file}\n")
    
    for i, config in enumerate(configs, 1):
        proj_req = "Auto (Advisor)" if config["proj"] is None else f"Forced_{config['proj']}"
        print(f"--- Run {i}/{len(configs)} | Config: {config['label']} | Thresh: {config['thresh']} | Proj: {proj_req} ---")
        
        try:
            result = safe_semantic_join(
                table_a, table_b,
                predicate=predicate_str,
                schema_a=["transcription"], 
                schema_b=["specialty"],
                llm_model="gpt-4o", 
                clustering="kmeans",
                n_clusters=4,             
                embedding=emb,
                filter_threshold=config["thresh"], 
                block_size=15,            
                cluster_size_limit=500,
                min_profile_size=2,       
                filter_sample_size=5,
                force_projection=config["proj"], 
                verbose=True
            )
            
            # Metrics
            pred = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
            tp = len(gt_pairs & pred)
            fp = len(pred - gt_pairs)
            fn = len(gt_pairs - pred)
            recall = (tp / len(gt_pairs) * 100) if gt_pairs else 0.0
            precision = (tp / len(pred) * 100) if pred else 0.0
            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
            
            # Extract internal engine state
            actual_strategy = result.plan.join_strategy
            used_projection_actual = "project" in result.timings
            
            # Calculate dropped clusters
            initial_pairs = set(result.cluster_pairs_initial or [])
            surviving_pairs = set(result.cluster_pairs_surviving or [])
            dropped_cluster_pairs = len(initial_pairs) - len(surviving_pairs)
            
            run_data = {
                "Run_ID": i,
                "Config_Name": config["label"],
                "Filter_Threshold": config["thresh"],
                "Projection_Requested": proj_req,
                "Projection_Executed": used_projection_actual,
                "Join_Strategy": actual_strategy,
                "Dropped_Clusters": dropped_cluster_pairs,
                "Recall": round(recall, 2),
                "Precision": round(precision, 2),
                "F1": round(f1, 2),
                "Tokens": result.tokens.total,
                "Time_s": round(sum(result.timings.values()), 2)
            }
            results.append(run_data)
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            print(f"  -> R:{recall:.1f}% P:{precision:.1f}% | Tokens:{result.tokens.total:,} | Dropped Clusters: {dropped_cluster_pairs}")
            time.sleep(2)
            
        except Exception as e:
            print(f"  -> Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_medical_search()