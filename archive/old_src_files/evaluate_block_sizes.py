import pandas as pd
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_block_join_no_cot(block_size, trial_num):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Load a random 'block_size' sample from each table using trial_num as the seed
    # Resetting the index ensures enumerate() and iterrows() align for ID generation
    df_a = pd.read_csv('data/table_a.csv').sample(n=block_size, random_state=trial_num).reset_index(drop=True)
    df_b = pd.read_csv('data/table_b.csv').sample(n=block_size, random_state=trial_num + 100).reset_index(drop=True)

    # 2. Establish Ground Truth (Matching Sentiments)
    expected_matches = []
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                expected_matches.append((f"A-{i}", f"B-{j}"))
    
    total_possible_combinations = block_size * block_size
    
    print(f"\n==================================================")
    print(f"--- Verification Run: Block Size {block_size}x{block_size} | Trial {trial_num + 1} ---")
    print(f"Ground Truth: Found {len(expected_matches)} matching pairs out of {total_possible_combinations} possible.")

    # 3. Construct the Block Join Prompt
    block_a = "\n".join([f"ID A-{i}: {r[:400]}..." for i, r in enumerate(df_a['review'])])
    block_b = "\n".join([f"ID B-{j}: {r[:400]}..." for j, r in enumerate(df_b['review'])])

    prompt = f"""
        You are a high-precision data joiner. Your task is to find all pairs of movie reviews that have the SAME sentiment (both positive or both negative).
        
        Compare EVERY item in TABLE A against EVERY item in TABLE B. 
        
        TABLE A:
        {block_a}
        
        TABLE B:
        {block_b}
        
        Return ONLY a JSON dictionary where each key is a Table A ID, and its value is a list of all matching Table B IDs. 
        If a Table A ID has no matches, return an empty list for that key. Do not provide explanations.
        
        Example format:
        {{
            "matches": {{
                "A-0": ["B-1", "B-5", "B-9"],
                "A-1": [],
                "A-2": ["B-2"]
            }}
        }}
        """

    # 4. Call LLM and Track Time
    print(f"Calling LLM for semantic matching...")
    
    start_time = time.time()
    try:
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

        # 5. Compare Results 
        raw_output = json.loads(response.choices[0].message.content)
        matches_dict = raw_output.get("matches", {})
        
        # Unwrap the dictionary back into a flat list of tuples
        llm_matches = []
        for a_id, b_list in matches_dict.items():
            for b_id in b_list:
                llm_matches.append((a_id, b_id))
                
    except Exception as e:
        print(f"Error during LLM call or parsing: {e}")
        llm_matches = []
        elapsed_time = time.time() - start_time
    
    expected_set = set(expected_matches)
    llm_set = set(llm_matches)

    true_positives = len(expected_set.intersection(llm_set))
    false_positives = len(llm_set - expected_set)
    false_negatives = len(expected_set - llm_set)

    # 6. Output Metrics
    recall = (true_positives / len(expected_set)) * 100 if expected_set else 0
    precision = (true_positives / len(llm_set)) * 100 if llm_set else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- Results for Trial {trial_num + 1} ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Recall (Accuracy): {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")
    
    return {
        "Time (s)": elapsed_time,
        "Total Tokens": total_prompt_tokens + total_completion_tokens,
        "Recall (%)": recall,
        "Precision (%)": precision,
        "F1 Score": f1_score
    }

if __name__ == "__main__":
    test_sizes = [5, 10, 15, 20, 25]
    num_trials = 3
    summary_results = []
    
    for size in test_sizes:
        size_metrics = []
        for trial in range(num_trials):
            metrics = run_block_join_no_cot(size, trial_num=trial)
            size_metrics.append(metrics)
            
        # Calculate averages across the 3 trials
        avg_time = sum(m["Time (s)"] for m in size_metrics) / num_trials
        avg_tokens = sum(m["Total Tokens"] for m in size_metrics) / num_trials
        avg_recall = sum(m["Recall (%)"] for m in size_metrics) / num_trials
        avg_precision = sum(m["Precision (%)"] for m in size_metrics) / num_trials
        avg_f1 = sum(m["F1 Score"] for m in size_metrics) / num_trials
        
        summary_results.append({
            "Block Size": f"{size}x{size}",
            "Avg Time (s)": round(avg_time, 2),
            "Avg Total Tokens": round(avg_tokens),
            "Avg Recall (%)": round(avg_recall, 2),
            "Avg Precision (%)": round(avg_precision, 2),
            "Avg F1 Score": round(avg_f1, 2)
        })
        
    print("\n\n==================================================")
    print("FINAL SUMMARY REPORT (Averages across 3 trials)")
    print("==================================================")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))