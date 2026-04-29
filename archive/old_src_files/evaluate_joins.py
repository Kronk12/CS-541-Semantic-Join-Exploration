import pandas as pd
import os

def clean_id(val):
    """Removes 'A-' or 'B-' prefixes so we can compare raw integers."""
    return int(str(val).replace('A-', '').replace('B-', '').strip())

def evaluate_ablation_study():
    print("==================================================")
    print("--- Final Pipeline Evaluation ---")
    print("==================================================\n")
    
    # 1. Establish Absolute Ground Truth from the EXACT clustered tables
    if not os.path.exists('data/table_a_clustered_distilbert.csv') or not os.path.exists('data/table_b_clustered_distilbert.csv'):
        print("Error: Clustered data files not found. Cannot establish ground truth.")
        return

    table_a = pd.read_csv('data/table_a_clustered_distilbert.csv')
    table_b = pd.read_csv('data/table_b_clustered_distilbert.csv')
    
    ground_truth = set()
    for i, row_a in table_a.iterrows():
        for j, row_b in table_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                ground_truth.add((i, j))
                
    total_expected = len(ground_truth)
    total_combinations = len(table_a) * len(table_b)
    print(f"Dataset Size: {len(table_a)}x{len(table_b)} ({total_combinations} pairs)")
    print(f"Absolute Ground Truth: {total_expected} matching pairs.")

    # 2. Pipeline Configurations to Evaluate
    configs = [
        ("Cluster Join (Filtered)", "data/final_matches_filtered_clusters.csv"),
        ("Cluster Join (All Clusters)", "data/final_matches_all_clusters.csv"),
        # You can uncomment and add any saved block join CSVs below to compare them directly!
        # ("Block Join (10x10 Adjacency)", "data/final_matches_block_10x10.csv")
    ]

    results = []

    for config_name, file_path in configs:
        if not os.path.exists(file_path):
            print(f"  -> Skipping {config_name}: File '{file_path}' not found.")
            continue
            
        block_df = pd.read_csv(file_path)
        
        predicted = set()
        for _, row in block_df.iterrows():
            try:
                id_a = clean_id(row['Table_A_ID'])
                id_b = clean_id(row['Table_B_ID'])
                predicted.add((id_a, id_b))
            except ValueError:
                pass
                
        # 3. Calculate Confusion Matrix Elements
        true_positives = ground_truth.intersection(predicted)
        false_positives = predicted - ground_truth
        false_negatives = ground_truth - predicted

        tp = len(true_positives)
        fp = len(false_positives)

        # 4. Calculate Final Metrics
        recall = (tp / total_expected) * 100 if total_expected > 0 else 0
        precision = (tp / len(predicted)) * 100 if len(predicted) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "Configuration": config_name,
            "Recall (%)": round(recall, 2),
            "Precision (%)": round(precision, 2),
            "F1-Score": round(f1_score, 2),
            "True Positives": tp,
            "Hallucinations": fp
        })

    # 5. Print Summary Table
    if results:
        print("\nFINAL END-TO-END PIPELINE RESULTS:")
        print("-" * 105)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        print("-" * 105)
    else:
        print("\nNo results to display. Run the join scripts to generate the CSV files.")

if __name__ == "__main__":
    evaluate_ablation_study()