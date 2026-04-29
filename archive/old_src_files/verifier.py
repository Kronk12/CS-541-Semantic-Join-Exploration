import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_block_join_no_cot():
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Load 5 rows from each table
    df_a = pd.read_csv('data/table_a.csv').head(10)
    df_b = pd.read_csv('data/table_b.csv').head(10)

    # 2. Establish Ground Truth (Matching Sentiments)
    expected_matches = []
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                expected_matches.append((f"A-{i}", f"B-{j}"))
    
    print(f"--- Verification Run (Block Join - NO Chain-of-Thought) ---")
    print(f"Ground Truth: Found {len(expected_matches)} matching pairs out of 25 possible.")

    # 3. Construct the Block Join Prompt
    block_a = "\n".join([f"ID A-{i}: {r[:400]}..." for i, r in enumerate(df_a['review'])])
    block_b = "\n".join([f"ID B-{j}: {r[:400]}..." for j, r in enumerate(df_b['review'])])

    # Removed the "STEP 1/2" reasoning instructions and the "analysis" JSON key
    prompt = f"""
    You are a high-precision data joiner. Your task is to find all pairs of movie reviews that have the SAME sentiment (both positive or both negative).
    
    Compare EVERY item in TABLE A against EVERY item in TABLE B. Output EVERY SINGLE PAIR that shares the same sentiment. Do not skip any valid combinations. Do not provide any explanations or reasoning.
    
    TABLE A:
    {block_a}
    
    TABLE B:
    {block_b}
    
    Return a JSON object exactly matching this structure:
    {{
        "matches": [["A-0", "B-1"], ["A-2", "B-5"]]
    }}
    """

    # 4. Call LLM
    print("Calling LLM for semantic matching (Direct JSON)...")
    response = client.chat.completions.create(
        model="gpt-5-mini", 
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" },
        # temperature=0 # Absolute determinism
    )

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    # 5. Compare Results
    raw_output = json.loads(response.choices[0].message.content)
    
    llm_matches = [tuple(pair) for pair in raw_output.get("matches", [])]
    
    expected_set = set(expected_matches)
    llm_set = set(llm_matches)

    true_positives = len(expected_set.intersection(llm_set))
    false_positives = len(llm_set - expected_set)
    false_negatives = len(expected_set - llm_set)

    # 6. Output Metrics
    accuracy = (true_positives / len(expected_set)) * 100 if expected_set else 0
    
    print(f"\n--- Results ---")
    print(f"Matches Predicted by LLM: {len(llm_matches)}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives (Hallucinations): {false_positives}")
    print(f"False Negatives (Misses): {false_negatives}")
    print(f"Recall (Accuracy): {accuracy:.2f}%")

    print(f"\n--- Token Usage & Cost Metrics ---")
    print(f"Total Input (Prompt) Tokens: {total_prompt_tokens}")
    print(f"Total Output (Completion) Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")

if __name__ == "__main__":
    run_block_join_no_cot()