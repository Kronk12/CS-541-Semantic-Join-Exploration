import json
import os
import pandas as pd

# Update these paths if your script is executed from a different directory
LOG_DIR = "src/evaluation/sim_logs"
DATA_DIR = "data"  
OUTPUT_CSV_DIR = "src/evaluation/logs" # Directory to store the organized CSVs

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
    
    gt_mapping_path = os.path.join(DATA_DIR, 'stack_ground_truth.csv')
    if not os.path.exists(gt_mapping_path):
        raise FileNotFoundError(f"Missing ground truth file: {gt_mapping_path}")
        
    df_gt = pd.read_csv(gt_mapping_path)
    valid_pairs = set(zip(df_gt['question_id'].astype(int), df_gt['concept_id'].astype(str)))
    
    gt = set()
    for i, row_a in df_a.iterrows():
        q_id = row_a.get("question_id")
        if pd.isna(q_id):
            continue
        q_id = int(q_id)
            
        for j, row_b in df_b.iterrows():
            c_id = str(row_b.get("concept_id"))
            if (q_id, c_id) in valid_pairs:
                gt.add((i, j))
                
    return gt

def get_imdb_ground_truth():
    """Calculates the ground truth directly from the IMDB CSVs."""
    df_a = pd.read_csv(os.path.join(DATA_DIR, 'table_a.csv'))
    df_b = pd.read_csv(os.path.join(DATA_DIR, 'table_b.csv'))
    
    gt = set()
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if row_a.get('sentiment') == row_b.get('sentiment') and pd.notna(row_a.get('sentiment')):
                gt.add((i, j))
    return gt

def evaluate_subgroup(dataset_base_name, use_projection, ratios, thresholds, ground_truth):
    """Parses JSON logs for a specific subgroup, simulates outcomes, and saves to CSV."""
    results = []
    suffix = "_projection" if use_projection else ""
    subgroup_name = f"{dataset_base_name}{suffix}"
    
    for ratio in ratios:
        filename = f"{dataset_base_name}_master_log_ratio_{ratio}{suffix}.json"
        filepath = os.path.join(LOG_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  [!] Missing file: {filename} (Skipping)")
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
                if pair["sample_rate"] >= thresh:
                    simulated_matches.update(tuple(m) for m in pair["matches"])
                    total_prompt += pair["tokens"]["prompt"]
                    total_comp += pair["tokens"]["completion"]
                    pairs_kept += 1
                else:
                    pairs_dropped += 1
                    
            cost = (total_prompt * 2.50 / 1_000_000) + (total_comp * 10.00 / 1_000_000)
            tp = len(ground_truth & simulated_matches)
            
            recall = (tp / len(ground_truth) * 100) if ground_truth else 0.0
            precision = (tp / len(simulated_matches) * 100) if simulated_matches else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            
            results.append({
                "Subgroup": subgroup_name,
                "Ratio": ratio,
                "Threshold": thresh,
                "Kept Pairs": pairs_kept,
                "Dropped Pairs": pairs_dropped,
                "Recall (%)": round(recall, 2),
                "Precision (%)": round(precision, 2),
                "F1 (%)": round(f1, 2),
                "Total Tokens": total_prompt + total_comp,
                "Cost ($)": round(cost, 4)
            })
            
    if results:
        df = pd.DataFrame(results)
        os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
        out_filename = f"{subgroup_name}_aggregated_results.csv"
        out_path = os.path.join(OUTPUT_CSV_DIR, out_filename)
        df.to_csv(out_path, index=False)
        print(f"  -> Saved {len(results)} threshold simulations to {out_filename}")
    else:
        print(f"  -> No data found to save for {subgroup_name}")

if __name__ == "__main__":
    # Define the simulation bounds
    ratios = [0.025, 0.05, 0.075, 0.1]
    
    # Generate thresholds from 0.0 to 1.00 in increments of 0.01
    thresholds = [round(x * 0.01, 2) for x in range(101)]
    
    print("Calculating Ground Truths...")
    gt_emails = get_emails_ground_truth()
    gt_stack = get_stackoverflow_ground_truth()
    gt_imdb = get_imdb_ground_truth()
    
    print(f"  Emails GT: {len(gt_emails)} pairs")
    print(f"  StackOverflow GT: {len(gt_stack)} pairs")
    print(f"  IMDB GT: {len(gt_imdb)} pairs\n")
    
    # Map the base JSON filenames to their respective ground truth sets
    subgroups_to_run = [
        ("emails", gt_emails),
        ("stackoverflow", gt_stack),
        ("stackoverflow_no_desc", gt_stack),
        ("imdb", gt_imdb)
    ]
    
    print("="*60)
    print("STARTING SIMULATION EVALUATIONS")
    print("="*60)
    
    for base_name, gt in subgroups_to_run:
        print(f"\nProcessing Group: {base_name.upper()}")
        for use_proj in [False, True]:
            evaluate_subgroup(base_name, use_proj, ratios, thresholds, gt)
            
    print("\n" + "="*60)
    print(f"Done! All CSVs have been exported to {OUTPUT_CSV_DIR}/")
    print("="*60)