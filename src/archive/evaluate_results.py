import pandas as pd
import os

def clean_id(val):
    """Removes 'A-' or 'B-' prefixes so we can compare raw integers."""
    return str(val).replace('A-', '').replace('B-', '').strip()

def evaluate_pipeline():
    print("--- Semantic Join Evaluation ---")
    
    # 1. Load the Data
    if not os.path.exists('data/naive_results.csv') or not os.path.exists('data/final_matches.csv'):
        print("Error: Missing result files. Make sure you ran both the naive join and block join.")
        return

    naive_df = pd.read_csv('data/naive_results.csv')
    block_df = pd.read_csv('data/final_matches.csv')

    # 2. Clean the IDs to ensure a 1:1 match
    naive_df['id_a'] = naive_df['id_a'].apply(clean_id)
    naive_df['id_b'] = naive_df['id_b'].apply(clean_id)
    
    block_df['Table_A_ID'] = block_df['Table_A_ID'].apply(clean_id)
    block_df['Table_B_ID'] = block_df['Table_B_ID'].apply(clean_id)

    # 3. Create Sets of Tuples for easy comparison
    ground_truth = set(zip(naive_df['id_a'], naive_df['id_b']))
    predicted = set(zip(block_df['Table_A_ID'], block_df['Table_B_ID']))

    print(f"Total Matches Expected (Naive Ground Truth): {len(ground_truth)}")
    print(f"Total Matches Found (Block Join Pipeline): {len(predicted)}\n")

    # 4. Calculate Confusion Matrix Elements
    true_positives = ground_truth.intersection(predicted)
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    # 5. Calculate Metrics
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 6. Output the Final Report for the Presentation
    print("--- Accuracy Metrics ---")
    print(f"True Positives (Correct Matches)     : {tp}")
    print(f"False Positives (Hallucinations)     : {fp}")
    print(f"False Negatives (Missed Matches)     : {fn}\n")
    
    print(f"Recall (Did we find them all?)       : {recall:.2f}%")
    print(f"Precision (Are our matches correct?) : {precision:.2f}%")
    print(f"F1-Score (Overall Balance)           : {f1_score:.2f}%\n")

    if false_negatives:
        print("Sample of Missed Matches (FN):")
        for pair in list(false_negatives)[:3]:
            print(f"  Table A ID: {pair[0]} <--> Table B ID: {pair[1]}")

if __name__ == "__main__":
    evaluate_pipeline()