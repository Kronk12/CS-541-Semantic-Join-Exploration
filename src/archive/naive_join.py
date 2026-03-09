import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_naive_join_no_cot():
    total_prompt_tokens = 0
    total_completion_tokens = 0
    df_a = pd.read_csv('data/table_a.csv').head(5)
    df_b = pd.read_csv('data/table_b.csv').head(5)

    expected_matches = []
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            if row_a['sentiment'] == row_b['sentiment']:
                expected_matches.append((i, j))
    
    print(f"--- Naive Join (NO Chain-of-Thought) ---")
    print(f"Ground Truth: {len(expected_matches)} expected matches.")

    llm_matches = []
    for i, row_a in df_a.iterrows():
        for j, row_b in df_b.iterrows():
            # Stripped-down Binary Prompt
            prompt = f"""
            Task: Compare the sentiment of two movie reviews.
            
            Review A: "{row_a['review'][:300]}"
            Review B: "{row_b['review'][:300]}"
            
            Do they have the SAME sentiment (both positive or both negative)?
            Answer only YES or NO.
            """
            
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                # temperature=0,
                # max_tokens=2 # Strictly enforces the binary output to save tokens
            )

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            output = response.choices[0].message.content.strip().upper()
            if "YES" in output:
                llm_matches.append((i, j))

    # Metrics
    true_positives = len(set(expected_matches).intersection(set(llm_matches)))
    recall = (true_positives / len(expected_matches)) * 100 if expected_matches else 0
    print(f"\nNew Naive Recall (No CoT): {recall:.2f}% ({true_positives}/{len(expected_matches)})")

    print(f"\n--- Token Usage & Cost Metrics ---")
    print(f"Total Input (Prompt) Tokens: {total_prompt_tokens}")
    print(f"Total Output (Completion) Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")

if __name__ == "__main__":
    run_naive_join_no_cot()