import os
import sys
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

from semantic_join import semantic_join

def get_emails_ground_truth(df_a, df_b):
    gt = set()
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if (row_a["name"] == row_b["name"]) and (row_b["month_idx"] < row_a["month_idx"]):
                gt.add((i, j))
    return gt

def run_trial(df_a, df_b, labels, gt, use_projection, run_id):
    proj_label = "with projection" if use_projection else "no projection"
    print(f"\n{'='*60}")
    print(f"Classifier Join — {proj_label}")
    print(f"{'='*60}")

    start_time = time.time()

    result = semantic_join(
        table_a=df_a,
        table_b=df_b,
        predicate="The texts refer to the exact same person, and the internal email in Table B proves the witness statement in Table A is a lie.",
        schema_a=["statement"],
        schema_b=["email"],
        force_strategy="classifier",
        force_labels=labels,
        force_projection=use_projection,
        verbose=False
    )

    time_s = time.time() - start_time

    predicted_matches = set(zip(result.matches["a_idx"], result.matches["b_idx"]))
    tp = len(gt & predicted_matches)
    recall = (tp / len(gt) * 100) if gt else 0.0
    precision = (tp / len(predicted_matches) * 100) if predicted_matches else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total_tokens = result.tokens.prompt_tokens + result.tokens.completion_tokens

    print(f"Recall: {recall:.2f}%  Precision: {precision:.2f}%  F1: {f1:.2f}%  Tokens: {total_tokens:,}  Time: {time_s:.1f}s")

    baseline_type = "Classifier_Join_Projection" if use_projection else "Classifier_Join"
    return f"{run_id},Emails,1,{baseline_type},{len(df_a)},{len(df_b)},{len(df_a)*len(df_b)},{recall:.2f},{precision:.2f},{f1:.2f},{total_tokens},{time_s:.2f}\n"

def evaluate_emails():
    df_a = pd.read_csv('data/table_a_emails.csv')
    df_b = pd.read_csv('data/table_b_emails.csv')
    labels = list(df_b['name'].unique()) + ['unknown']

    print("Calculating Ground Truth...")
    gt = get_emails_ground_truth(df_a, df_b)
    print(f"GT size: {len(gt)} pairs")

    out_path = 'src/results/cluster_join_emails_classification.csv'
    rows = []
    rows.append(run_trial(df_a, df_b, labels, gt, use_projection=False, run_id=1))
    rows.append(run_trial(df_a, df_b, labels, gt, use_projection=True, run_id=2))

    with open(out_path, "w") as f:
        f.write("Run_ID,Dataset,Trial,Baseline_Type,A_Size,B_Size,Total_Pairs_Evaluated,Recall,Precision,F1,Tokens,Time_s\n")
        f.writelines(rows)

    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    evaluate_emails()
