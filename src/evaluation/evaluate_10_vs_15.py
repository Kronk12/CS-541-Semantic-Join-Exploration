import json
import os
import pandas as pd

LOG_DIR = "src/evaluation/sim_logs"
DATA_DIR = "data"

def get_stackoverflow_ground_truth():
    df_a = pd.read_csv(os.path.join(DATA_DIR, 'table_a_stack.csv'))
    df_b = pd.read_csv(os.path.join(DATA_DIR, 'table_b_stack.csv'))
    
    gt_mapping_path = os.path.join(DATA_DIR, 'stack_ground_truth.csv')
    df_gt = pd.read_csv(gt_mapping_path)
    valid_pairs = set(zip(df_gt['question_id'].astype(int), df_gt['concept_id'].astype(str)))
    
    gt = set()
    for i, row_a in df_a.iterrows():
        q_id = row_a.get("question_id")
        if pd.isna(q_id): continue
        q_id = int(q_id)
            
        for j, row_b in df_b.iterrows():
            c_id = str(row_b.get("concept_id"))
            if (q_id, c_id) in valid_pairs:
                gt.add((i, j))
    return gt

def evaluate_block_10_vs_15():
    gt_stack = get_stackoverflow_ground_truth()
    ratios = [0.025, 0.05]
    thresholds = [round(x * 0.01, 2) for x in range(101)]
    
    results = []
    
    for ratio in ratios:
        # 1. Path to the NEW Block 10 files you just generated
        filepath_10 = os.path.join(LOG_DIR, f"stackoverflow_no_desc_master_log_ratio_10_{ratio}_projection.json")
        
        # 2. Path to the OLD Block 15 files from your previous run
        filepath_15 = os.path.join(LOG_DIR, f"stackoverflow_no_desc_master_log_ratio_{ratio}_projection.json")
        
        for block_size, filepath in [(10, filepath_10), (15, filepath_15)]:
            if not os.path.exists(filepath):
                print(f"Missing file: {filepath}")
                continue
                
            with open(filepath, "r") as f:
                data = json.load(f)
                
            base_prompt = data["fixed_tokens"]["prompt"]
            base_comp = data["fixed_tokens"]["completion"]
            
            for thresh in thresholds:
                simulated_matches = set()
                total_prompt = base_prompt
                total_comp = base_comp
                
                for pair in data["pairs"]:
                    if pair["sample_rate"] >= thresh:
                        simulated_matches.update(tuple(m) for m in pair["matches"])
                        total_prompt += pair["tokens"]["prompt"]
                        total_comp += pair["tokens"]["completion"]
                        
                cost = (total_prompt * 2.50 / 1_000_000) + (total_comp * 10.00 / 1_000_000)
                tp = len(gt_stack & simulated_matches)
                
                recall = (tp / len(gt_stack) * 100) if gt_stack else 0.0
                precision = (tp / len(simulated_matches) * 100) if simulated_matches else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                
                results.append({
                    "Block_Size": block_size,
                    "Ratio": ratio,
                    "Threshold": thresh,
                    "Recall (%)": round(recall, 2),
                    "Precision (%)": round(precision, 2),
                    "F1 (%)": round(f1, 2),
                    "Total Tokens": total_prompt + total_comp,
                    "Cost ($)": round(cost, 4)
                })
                
    if results:
        df = pd.DataFrame(results)
        # Find the best F1 for each Block Size and Ratio combo
        idx = df.groupby(['Block_Size', 'Ratio'])['F1 (%)'].idxmax()
        best_f1 = df.loc[idx, ['Block_Size', 'Ratio', 'Threshold', 'F1 (%)', 'Recall (%)', 'Precision (%)', 'Cost ($)']]
        
        print("\n=== OPTIMAL F1: BLOCK 10 vs BLOCK 15 ===")
        print(best_f1.sort_values(by=['Ratio', 'Block_Size']).to_string(index=False))
        
        print("\n=== BASELINE (NO FILTERING, THRESHOLD = 0.0) ===")
        baseline = df[df['Threshold'] == 0.0][['Block_Size', 'Ratio', 'F1 (%)', 'Recall (%)', 'Precision (%)', 'Cost ($)']]
        print(baseline.sort_values(by=['Ratio', 'Block_Size']).to_string(index=False))

if __name__ == "__main__":
    evaluate_block_10_vs_15()