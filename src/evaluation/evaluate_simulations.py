import json
import os
import pandas as pd

# Update these paths if your script is executed from a different directory
LOG_DIR = "src/evaluation/sim_logs"
DATA_DIR = "data"  # Assuming data is at the root level

def get_emails_ground_truth():
    """Calculates the ground truth directly from the Emails CSVs."""
    df_a = pd.read_csv(os.path.join(DATA_DIR, 'table_a_emails.csv'))
    df_b = pd.read_csv(os.path.join(DATA_DIR, 'table_b_emails.csv'))
    
    gt = set()
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            # Predicate: Contradiction (same person, earlier month contradicting later month)
            if (row_a["name"] == row_b["name"]) and (row_b["month_idx"] < row_a["month_idx"]):
                gt.add((i, j))
    return gt

def get_stackoverflow_ground_truth():
    """Calculates the ground truth directly from the StackOverflow CSVs and the GT mapping file."""
    df_a = pd.read_csv(os.path.join(DATA_DIR, 'table_a_stack.csv'))
    df_b = pd.read_csv(os.path.join(DATA_DIR, 'table_b_stack.csv'))
    
    # Load the ground truth mapping file
    gt_mapping_path = os.path.join(DATA_DIR, 'stack_ground_truth.csv')
    if not os.path.exists(gt_mapping_path):
        raise FileNotFoundError(f"Missing ground truth file: {gt_mapping_path}")
        
    df_gt = pd.read_csv(gt_mapping_path)
    
    # Create a fast-lookup set of the valid pairs
    # Force types to ensure reliable matching (int for question_id, string for concept_id)
    valid_pairs = set(zip(df_gt['question_id'].astype(int), df_gt['concept_id'].astype(str)))
    
    gt = set()
    
    for i, row_a in df_a.iterrows():
        # Safely get the question_id
        q_id = row_a.get("question_id")
        if pd.isna(q_id):
            continue
        q_id = int(q_id)
            
        for j, row_b in df_b.iterrows():
            # Safely get the concept_id
            c_id = str(row_b.get("concept_id"))
            
            # If this specific combination exists in your ground truth file, record the row indices
            if (q_id, c_id) in valid_pairs:
                gt.add((i, j))
                
    return gt

def evaluate_logs(dataset_name, ratios, thresholds, ground_truth):
    """Parses the JSON logs for a specific dataset and simulates outcomes across different thresholds."""
    results = []
    
    for ratio in ratios:
        # Loop through both standard and projection logs
        for use_projection in [False, True]:
            suffix = "_projection" if use_projection else ""
            filename = f"{dataset_name}_master_log_ratio_{ratio}{suffix}.json"
            filepath = os.path.join(LOG_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"Skipping missing file: {filepath}")
                continue
                
            with open(filepath, "r") as f:
                data = json.load(f)
                
            base_prompt = data["fixed_tokens"]["prompt"]
            base_comp = data["fixed_tokens"]["completion"]
            
            for thresh in thresholds:
                simulated_matches = set()
                total_prompt = base_prompt
                total_comp = base_comp
                pairs_kept = 0
                pairs_dropped = 0
                
                for pair in data["pairs"]:
                    # If sample_rate passes the threshold, we "pay" for the tokens and keep matches
                    if pair["sample_rate"] >= thresh:
                        simulated_matches.update(tuple(m) for m in pair["matches"])
                        total_prompt += pair["tokens"]["prompt"]
                        total_comp += pair["tokens"]["completion"]
                        pairs_kept += 1
                    else:
                        pairs_dropped += 1
                        
                # Cost calculation (using GPT-4o pricing: $2.50/1M input, $10.00/1M output)
                cost = (total_prompt * 2.50 / 1_000_000) + (total_comp * 10.00 / 1_000_000)
                
                # Accuracy metrics
                tp = len(ground_truth & simulated_matches)
                
                recall = (tp / len(ground_truth) * 100) if ground_truth else 0.0
                precision = (tp / len(simulated_matches) * 100) if simulated_matches else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                
                results.append({
                    "Dataset": dataset_name.capitalize(),
                    "Ratio": ratio,
                    "Projection": use_projection,
                    "Threshold": thresh,
                    "Kept Pairs": pairs_kept,
                    "Dropped Pairs": pairs_dropped,
                    "Recall (%)": round(recall, 1),
                    "Precision (%)": round(precision, 1),
                    "F1 (%)": round(f1, 1),
                    "Total Tokens": total_prompt + total_comp,
                    "Cost ($)": round(cost, 4)
                })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define the simulation bounds
    ratios = [0.025, 0.05, 0.075, 0.1]
    
    # Test thresholds from 0.0 (no filtering) up to an aggressive 0.45
    thresholds = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45]
    
    all_results = []
    
    # --- EVALUATE EMAILS ---
    print("\nCalculating Ground Truth for Emails...")
    gt_emails = get_emails_ground_truth()
    print(f"Total True Matches (Emails): {len(gt_emails)}")
    
    print("Evaluating Emails Simulations...")
    df_emails = evaluate_logs("emails", ratios, thresholds, gt_emails)
    all_results.append(df_emails)

    # --- EVALUATE STACKOVERFLOW ---
    print("\nCalculating Ground Truth for StackOverflow...")
    gt_stack = get_stackoverflow_ground_truth()
    print(f"Total True Matches (StackOverflow): {len(gt_stack)}")
    
    print("Evaluating StackOverflow Simulations...")
    df_stack = evaluate_logs("stackoverflow", ratios, thresholds, gt_stack)
    all_results.append(df_stack)
    
    # Combine and sort all results
    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        # Sort so comparison is easy to read
        df_final = df_final.sort_values(by=["Dataset", "Ratio", "Threshold", "Projection"])
        
        print("\n" + "="*80)
        print("FINAL SIMULATION RESULTS")
        print("="*80)
        print(df_final.to_string(index=False))
    else:
        print("No simulation data found.")