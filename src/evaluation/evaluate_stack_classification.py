import os
import sys
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

from semantic_join import semantic_join

def get_stack_ground_truth(df_a, df_b):
    df_gt = pd.read_csv('data/stack_ground_truth.csv')
    valid_pairs = set(zip(df_gt['question_id'].astype(int), df_gt['concept_id'].astype(str)))
    gt = set()
    for i, row_a in df_a.iterrows():
        q_id = row_a.get("question_id")
        if pd.isna(q_id):
            continue
        for j, row_b in df_b.iterrows():
            if (int(q_id), str(row_b.get("concept_id"))) in valid_pairs:
                gt.add((i, j))
    return gt

def evaluate_stack():
    df_a = pd.read_csv('data/table_a_stack.csv')
    df_b = pd.read_csv('data/table_b_stack.csv')

    labels = list(df_b['concept_name'].astype(str)) + ['unknown']

    print("\n" + "="*60)
    print("Running Classifier Join evaluation for StackOverflow")
    print(f"Labels: {labels}")
    print("="*60)

    start_time = time.time()

    result = semantic_join(
        table_a=df_a,
        table_b=df_b,
        predicate="The question describes symptoms, errors, or intents that are solved by or directly related to this programming concept.",
        schema_a=["question_text"],
        schema_b=["concept_name"],
        force_strategy="classifier",
        force_labels=labels,
        force_projection=False,
        verbose=False
    )

    time_s = time.time() - start_time

    print("Calculating Ground Truth...")
    gt = get_stack_ground_truth(df_a, df_b)

    predicted_matches = set(zip(result.matches["a_idx"], result.matches["b_idx"]))

    tp = len(gt & predicted_matches)
    recall = (tp / len(gt) * 100) if gt else 0.0
    precision = (tp / len(predicted_matches) * 100) if predicted_matches else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    total_tokens = result.tokens.prompt_tokens + result.tokens.completion_tokens

    print(f"\nRecall: {recall:.2f}%  Precision: {precision:.2f}%  F1: {f1:.2f}%  Tokens: {total_tokens:,}  Time: {time_s:.1f}s")

    out_path = 'src/results/cluster_join_stack_classification.csv'
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a") as f:
        if not file_exists:
            f.write("Run_ID,Dataset,Trial,Baseline_Type,A_Size,B_Size,Total_Pairs_Evaluated,Recall,Precision,F1,Tokens,Time_s\n")
        f.write(f"1,StackOverflow,1,Classifier_Join,{len(df_a)},{len(df_b)},{len(df_a)*len(df_b)},{recall:.2f},{precision:.2f},{f1:.2f},{total_tokens},{time_s:.2f}\n")

    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_stack()
