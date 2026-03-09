import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_naive_metrics_no_cot():
    # 1. Setup and Token Trackers
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Keeping it 5x5 (25 comparisons)
    df_a = pd.read_csv('data/table_a.csv').head(10)
    df_b = pd.read_csv('data/table_b.csv').head(10)

    # 2. Establish Ground Truth
    expected_matches = []
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                expected_matches.append((i, j))
    
    print(f"\n==================================================")
    print(f"--- Naive Join Run: 5x5 Grid (NO Chain-of-Thought) ---")
    print(f"Ground Truth: Found {len(expected_matches)} matching pairs out of 25 possible.")
    print("Calling LLM for semantic matching (Pair-by-Pair)...")

    llm_matches = []
    
    # Start Timer
    start_time = time.time()
    
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            # Stripped-down Binary Prompt
            prompt = f"""
            Task: Compare the sentiment of two movie reviews.
            
            Review A: "{row_a['review'][:400]}"
            Review B: "{row_b['review'][:400]}"
            
            Do they have the SAME sentiment (both positive or both negative)?
            Answer only YES or NO.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o", # Keep the model consistent with your block test!
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2, # Strictly enforces the binary output to save completion tokens
                temperature=0
            )

            # Track tokens for this specific call
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
            output = response.choices[0].message.content.strip().upper()
            if "YES" in output:
                llm_matches.append((i, j))

    # End Timer
    elapsed_time = time.time() - start_time

    # 3. Calculate Confusion Matrix Elements
    expected_set = set(expected_matches)
    llm_set = set(llm_matches)

    true_positives = len(expected_set.intersection(llm_set))
    false_positives = len(llm_set - expected_set)
    false_negatives = len(expected_set - llm_set)

    # 4. Output Final Metrics
    accuracy = (true_positives / len(expected_set)) * 100 if expected_set else 0
    
    print(f"\n--- Results ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Matches Predicted by LLM: {len(llm_matches)}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives (Hallucinations): {false_positives}")
    print(f"False Negatives (Misses): {false_negatives}")
    print(f"Recall (Accuracy): {accuracy:.2f}%")

    print(f"\n--- Token Usage ---")
    print(f"Total Input (Prompt) Tokens: {total_prompt_tokens}")
    print(f"Total Output (Completion) Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_naive_metrics_no_cot()