import os
import sys
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

from semantic_join import semantic_join

def get_imdb_ground_truth(df_a, df_b):
    """Calculates the ground truth directly from the CSVs based on true labels."""
    gt = set()
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            # Update 'sentiment' to the actual column name in your CSV that holds the true label
            if row_a.get('sentiment') == row_b.get('sentiment') and pd.notna(row_a.get('sentiment')):
                gt.add((i, j))
    return gt

def evaluate_imdb():
    # Load your dataset
    df_a = pd.read_csv('data/table_a.csv')
    df_b = pd.read_csv('data/table_b.csv')

    print("\n" + "="*60)
    print("Running execution evaluation for IMDB")
    print("="*60)
    
    start_time = time.time()
    
    result = semantic_join(
        table_a=df_a,
        table_b=df_b,
        predicate="Both reviews express the same sentiment (Positive or Negative)",
        schema_a=["review"],
        schema_b=["review"],
        
        # --- FORCING PARAMETERS TO SKIP ADVISOR ---
        force_strategy="classifier",
        force_labels=['Positive', 'Negative', 'unknown'],
        force_projection=False,
        
        verbose=False # Set to false to keep output clean for the CSV record
    )

    time_s = time.time() - start_time

    # Calculate metrics
    print("Calculating Ground Truth...")
    gt = get_imdb_ground_truth(df_a, df_b)
    
    # Extract predicted matches from the DataFrame returned by semantic_join
    predicted_matches = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
    
    tp = len(gt & predicted_matches)
    recall = (tp / len(gt) * 100) if gt else 0.0
    precision = (tp / len(predicted_matches) * 100) if predicted_matches else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    total_tokens = result.tokens.prompt_tokens + result.tokens.completion_tokens

    log_dir = os.path.join(current_dir, "logs") # Adjust based on where evaluate_imdb.py lives
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, "cluster_join_imdb.csv")
    
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, "a") as f:
        if not file_exists:
            f.write("Run_ID,Dataset,Trial,Baseline_Type,A_Size,B_Size,Total_Pairs_Evaluated,Recall,Precision,F1,Tokens,Time_s\n")
        f.write(f"2,IMDB,1,Classifier_Join,{len(df_a)},{len(df_b)},{len(df_a)*len(df_b)},{recall:.2f},{precision:.2f},{f1:.2f},{total_tokens},{time_s:.2f}\n")
        
    print(f"\nResults appended to {filepath}")

if __name__ == "__main__":
    evaluate_imdb()