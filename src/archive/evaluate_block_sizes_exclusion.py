import pandas as pd
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_block_join_exclusion(block_size):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Load dynamic 'block_size' rows from each table
    df_a = pd.read_csv('data/table_a.csv').head(block_size)
    df_b = pd.read_csv('data/table_b.csv').head(block_size)

    # 2. Establish Ground Truth and All Possible Pairs
    expected_matches = set()
    all_possible_pairs = set()
    
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            pair_id = (f"A-{i}", f"B-{j}")
            all_possible_pairs.add(pair_id)
            if row_a['sentiment'] == row_b['sentiment']:
                expected_matches.add(pair_id)
    
    total_possible_combinations = block_size * block_size
    
    print(f"\n==================================================")
    print(f"--- Verification Run: Block Size {block_size}x{block_size} (EXCLUSION) ---")
    print(f"Ground Truth: Found {len(expected_matches)} matching pairs out of {total_possible_combinations} possible.")

    # 3. Construct the Exclusion Prompt
    block_a = "\n".join([f"ID A-{i}: {r[:400]}..." for i, r in enumerate(df_a['review'])])
    block_b = "\n".join([f"ID B-{j}: {r[:400]}..." for j, r in enumerate(df_b['review'])])

    prompt = f"""
        Find all pairs (Review A, Review B) that have DIFFERENT sentiments (One is Positive and the other is Negative).
        
        TABLE A:
        {block_a}
        
        TABLE B:
        {block_b}
        
        Return ONLY a JSON object with the key "non_matches".
        Example: {{"non_matches": [["A-ID", "B-ID"], ...]}}
        If all pairs have the same sentiment (0 non-matches), return {{"non_matches": []}}.
        """

    # 4. Call LLM and Track Time
    print(f"Calling LLM for semantic matching (Looking for exclusions)...")
    
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

    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    # 5. Parse Results and Invert Logic
    raw_output = json.loads(response.choices[0].message.content)
    
    llm_non_matches = {tuple(pair) for pair in raw_output.get("non_matches", [])}
    
    # Matches = All Pairs - Non-Matches
    llm_set = all_possible_pairs - llm_non_matches

    # 6. Calculate Confusion Matrix Elements
    true_positives = len(expected_matches.intersection(llm_set))
    false_positives = len(llm_set - expected_matches)
    false_negatives = len(expected_matches - llm_set)

    # 7. Output Metrics
    recall = (true_positives / len(expected_matches)) * 100 if expected_matches else 0
    precision = (true_positives / len(llm_set)) * 100 if llm_set else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- Results ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"LLM found {len(llm_non_matches)} non-matches, inferring {len(llm_set)} matches.")
    print(f"True Positives: {true_positives}")
    print(f"False Positives (Hallucinations/Missed Exclusions): {false_positives}")
    print(f"False Negatives (Incorrect Exclusions): {false_negatives}")
    print(f"Recall: {recall:.2f}% | Precision: {precision:.2f}%")

    print(f"\n--- Token Usage ---")
    print(f"Total Input (Prompt) Tokens: {total_prompt_tokens}")
    print(f"Total Output (Completion) Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    
    # Return metrics for the summary table
    return {
        "Block Size": f"{block_size}x{block_size}",
        "Time (s)": round(elapsed_time, 2),
        "Total Tokens": total_prompt_tokens + total_completion_tokens,
        "Recall (%)": round(recall, 2),
        "Precision (%)": round(precision, 2),
        "F1 Score": round(f1_score, 2)
    }

if __name__ == "__main__":
    test_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
    summary_results = []
    
    for size in test_sizes:
        metrics = run_block_join_exclusion(size)
        summary_results.append(metrics)
        
    print("\n\n==================================================")
    print("FINAL SUMMARY REPORT")
    print("==================================================")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))