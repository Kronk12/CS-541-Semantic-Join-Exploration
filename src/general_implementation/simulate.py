import json

import pandas as pd

def _get_cluster_metadata(df: pd.DataFrame, labels: list, num_samples: int) -> dict:
    """Extracts sizes and sample rows for each valid cluster."""
    if labels is None:
        return {}
        
    df_labeled = df.assign(_cluster=labels)
    valid_clusters = df_labeled[df_labeled["_cluster"] >= 0]
    
    meta = {}
    for cid, group in valid_clusters.groupby("_cluster"):
        sample_df = group.head(num_samples).drop(columns="_cluster")
        # Ensure values are JSON serializable
        meta[int(cid)] = {
            "size": len(group),
            "samples": sample_df.to_dict(orient="records")
        }
    return meta

def export_simulation_data(
    result, 
    table_a: pd.DataFrame, 
    table_b: pd.DataFrame, 
    filepath: str, 
    num_samples: int = 3
):
    """Exports cluster metadata, pair stats, and rates to JSON."""
    
    # Calculate fixed costs (tokens spent BEFORE Stage 4)
    # Total tokens minus the tokens spent on all Stage 4 joins
    stage_4_prompt = sum(p.tokens.prompt_tokens for p in result.cluster_pair_stats)
    stage_4_comp = sum(p.tokens.completion_tokens for p in result.cluster_pair_stats)
    
    base_prompt = result.tokens.prompt_tokens - stage_4_prompt
    base_comp = result.tokens.completion_tokens - stage_4_comp

    data = {
        "fixed_tokens": {
            "prompt": base_prompt,
            "completion": base_comp
        },
        "clusters_a": _get_cluster_metadata(table_a, result.labels_a, num_samples),
        "clusters_b": _get_cluster_metadata(table_b, result.labels_b, num_samples),
        "pairs": []
    }

    # Match rates were captured in cluster_filter.py
    rates = result.cluster_match_rates or {}

    for stat in result.cluster_pair_stats:
        pair_key = (stat.ca, stat.cb)
        
        # If a pair was skipped in sampling (e.g., under min_profile_size), 
        # it has no rate, meaning it bypasses the filter. We use a rate of 1.0 
        # so the simulator always includes it, matching pipeline behavior.
        rate = rates.get(pair_key, 1.0)

        data["pairs"].append({
            "ca": stat.ca,
            "cb": stat.cb,
            "size_a": stat.size_a,
            "size_b": stat.size_b,
            "sample_rate": rate,
            "matches": stat.matches,
            "tokens": {
                "prompt": stat.tokens.prompt_tokens,
                "completion": stat.tokens.completion_tokens
            }
        })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Simulation data exported to {filepath}")

def simulate_threshold(filepath: str, threshold: float, ground_truth: set = None):
    """Simulates pipeline performance at a specific threshold."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Start with the fixed costs (Embedding + Stage 3 Sampling)
    total_prompt = data["fixed_tokens"]["prompt"]
    total_comp = data["fixed_tokens"]["completion"]
    
    simulated_matches = set()
    pairs_kept = 0
    pairs_dropped = 0

    for pair in data["pairs"]:
        # Apply the threshold filter
        if pair["sample_rate"] >= threshold:
            # Pair passes: we 'pay' the tokens and 'collect' the matches
            simulated_matches.update(tuple(m) for m in pair["matches"])
            total_prompt += pair["tokens"]["prompt"]
            total_comp += pair["tokens"]["completion"]
            pairs_kept += 1
        else:
            # Pair dropped: we skip the matches and save the Stage 4 tokens
            pairs_dropped += 1

    # Calculate cost (Example using GPT-4o pricing)
    cost = (total_prompt * 2.50 / 1_000_000) + (total_comp * 10.00 / 1_000_000)

    print(f"--- Simulation Results (Threshold: {threshold}) ---")
    print(f"Clusters Kept : {pairs_kept}")
    print(f"Clusters Drop : {pairs_dropped}")
    print(f"Total Matches : {len(simulated_matches)}")
    print(f"Total Tokens  : {total_prompt + total_comp:,} "
          f"(P: {total_prompt:,} | C: {total_comp:,})")
    print(f"Est. Cost     : ${cost:.4f}")

    # If you pass a ground truth set, calculate accuracy metrics
    if ground_truth:
        tp = len(ground_truth & simulated_matches)
        fp = len(simulated_matches - ground_truth)
        fn = len(ground_truth - simulated_matches)
        
        recall = (tp / len(ground_truth) * 100) if ground_truth else 0.0
        precision = (tp / len(simulated_matches) * 100) if simulated_matches else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        
        print(f"\nRecall    : {recall:.1f}%")
        print(f"Precision : {precision:.1f}%")
        print(f"F1 Score  : {f1:.1f}%")
        
    return simulated_matches

# Example Usage:
# Run your pipeline with filter_threshold=-1 to generate master.json
# truth_set = {(1, 4), (2, 8), (3, 9)} # Extract from utils.ground_truth
# 
# simulate_threshold("master.json", threshold=0.05, ground_truth=truth_set)
# simulate_threshold("master.json", threshold=0.10, ground_truth=truth_set)