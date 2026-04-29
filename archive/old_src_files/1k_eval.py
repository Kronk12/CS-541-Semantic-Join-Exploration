import pandas as pd
from openai import OpenAI
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_quick_cluster_eval_exclusions(cluster_a_id, cluster_b_id, max_block_size=25):
    print(f"\n==================================================")
    print(f"--- Quick Eval (EXCLUSION): Cluster A-{cluster_a_id} vs Cluster B-{cluster_b_id} ---")
    print(f"==================================================")
    
    # 1. Load the 1k Data
    table_a = pd.read_csv('data/table_a_clustered_1k.csv')
    table_b = pd.read_csv('data/table_b_clustered_1k.csv')
    
    # 2. Extract and slice the clusters to a safe block size
    rows_a = table_a[table_a['cluster'] == cluster_a_id].head(max_block_size)
    rows_b = table_b[table_b['cluster'] == cluster_b_id].head(max_block_size)
    
    actual_size_a = len(rows_a)
    actual_size_b = len(rows_b)
    print(f"Processing Block Size: {actual_size_a} x {actual_size_b} ({actual_size_a * actual_size_b} combinations)")

    # 3. Establish Local Ground Truth & All Possible Pairs
    ground_truth_matches = set()
    all_possible_pairs = set()
    
    for idx_a, row_a in rows_a.iterrows():
        for idx_b, row_b in rows_b.iterrows():
            all_possible_pairs.add((idx_a, idx_b))
            if row_a['sentiment'] == row_b['sentiment']:
                ground_truth_matches.add((idx_a, idx_b))
                
    print(f"Ground Truth: Found {len(ground_truth_matches)} matching pairs out of {len(all_possible_pairs)}.")

    # 4. Format the EXCLUSION Prompt
    block_a_text = "\n".join([f"ID A-{idx}: {row['review'][:200]}" for idx, row in rows_a.iterrows()])
    block_b_text = "\n".join([f"ID B-{idx}: {row['review'][:200]}" for idx, row in rows_b.iterrows()])

    prompt = f"""
    Compare EVERY item in TABLE A against EVERY item in TABLE B. 
    Identify all pairs of reviews that have DIFFERENT sentiments (one is positive and the other is negative).
    
    TABLE A:
    {block_a_text}
    
    TABLE B:
    {block_b_text}
    
    Return a JSON object with a single key "non_matches" containing a list of ID pairs that DO NOT match in sentiment.
    Example: {{"non_matches": [["A-45", "B-12"], ["A-47", "B-15"]]}}
    If all pairs have the same sentiment (0 non-matches), return {{"non_matches": []}}.
    """

    # 5. Call GPT-4o
    print("Calling GPT-4o (Looking for exclusions)...")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" },
        temperature=0
    )
    
    elapsed_time = time.time() - start_time
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # 6. Parse Results and INVERT Logic
    llm_non_matches = set()
    try:
        result = json.loads(response.choices[0].message.content)
        for pair in result.get("non_matches", []):
            id_a = int(pair[0].replace('A-', ''))
            id_b = int(pair[1].replace('B-', ''))
            llm_non_matches.add((id_a, id_b))
    except Exception as e:
        print(f"Error parsing JSON: {e}")

    # The magic inversion: Matches = All Pairs - Non-Matches
    llm_matches = all_possible_pairs - llm_non_matches

    # 7. Calculate Metrics against Ground Truth MATCHES
    true_positives = ground_truth_matches.intersection(llm_matches)
    false_positives = llm_matches - ground_truth_matches
    false_negatives = ground_truth_matches - llm_matches

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    recall = (tp / len(ground_truth_matches)) * 100 if ground_truth_matches else 0
    precision = (tp / len(llm_matches)) * 100 if llm_matches else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 8. Output Report
    print(f"\n--- Results ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"LLM found {len(llm_non_matches)} non-matches, inferring {len(llm_matches)} matches.")
    print(f"True Positives: {tp}")
    print(f"False Positives (Hallucinations/Missed Exclusions): {fp}")
    print(f"False Negatives (Incorrect Exclusions): {fn}")
    print(f"Recall: {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")

    print(f"\n--- Token Usage ---")
    print(f"Total Tokens Used: {prompt_tokens + completion_tokens}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Test 1: Highly Pure Negative Clusters (Very few exclusions expected)
    run_quick_cluster_eval_exclusions(cluster_a_id=2, cluster_b_id=4, max_block_size=25)
    
    # Test 2: Highly Pure Positive Clusters (Very few exclusions expected)
    run_quick_cluster_eval_exclusions(cluster_a_id=15, cluster_b_id=14, max_block_size=25)
    
    # Test 3: Mixed/Messy Clusters (Lots of exclusions expected, testing the LLM's logic limit)
    run_quick_cluster_eval_exclusions(cluster_a_id=10, cluster_b_id=10, max_block_size=25)