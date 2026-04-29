import pandas as pd
import os

def clean_id(val):
    """Removes 'A-' or 'B-' prefixes so we can compare raw integers."""
    return int(str(val).replace('A-', '').replace('B-', '').strip())

def evaluate_block_join():
    print("--- Evaluating Block Join vs. Absolute Ground Truth ---")
    
    # 1. Establish Absolute Ground Truth from Source Files
    if not os.path.exists('data/table_a.csv') or not os.path.exists('data/table_b.csv'):
        print("Error: Missing source tables.")
        return
        
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')
    
    ground_truth = set()
    for i, row_a in table_a.iterrows():
        for j, row_b in table_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                ground_truth.add((i, j))
                
    print(f"Ground Truth: {len(ground_truth)} total possible matches.")

    # 2. Load Block Join Results
    if not os.path.exists('data/final_matches.csv'):
        print("Error: Missing final_matches.csv. Did the block join save successfully?")
        return

    block_df = pd.read_csv('data/final_matches.csv')
    
    predicted = set()
    for _, row in block_df.iterrows():
        try:
            id_a = clean_id(row['Table_A_ID'])
            id_b = clean_id(row['Table_B_ID'])
            predicted.add((id_a, id_b))
        except ValueError:
            pass # Skip rows with weird formatting if any exist
            
    print(f"Block Join Predictions: {len(predicted)} total matches proposed.\n")

    # 3. Calculate Confusion Matrix Elements
    true_positives = ground_truth.intersection(predicted)
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    # 4. Calculate Metrics
    recall = (tp / len(ground_truth)) * 100 if len(ground_truth) > 0 else 0
    precision = (tp / len(predicted)) * 100 if len(predicted) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 5. Output the Final Report
    print("--- Accuracy Metrics ---")
    print(f"True Positives (Correct Matches)     : {tp}")
    print(f"False Positives (Hallucinations)     : {fp}")
    print(f"False Negatives (Missed Matches)     : {fn}\n")
    
    print(f"Recall (Did we find them all?)       : {recall:.2f}%")
    print(f"Precision (Are our matches correct?) : {precision:.2f}%")
    print(f"F1-Score (Overall Balance)           : {f1_score:.2f}%\n")

if __name__ == "__main__":
    evaluate_block_join()