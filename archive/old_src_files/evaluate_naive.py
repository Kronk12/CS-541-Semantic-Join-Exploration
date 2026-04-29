import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_naive_metrics_no_cot():
    # 1. Load full datasets once
    full_df_a = pd.read_csv('data/table_a.csv')
    full_df_b = pd.read_csv('data/table_b.csv')
    
    num_trials = 5
    sample_size = 5
    # NEW: Define a base seed for reproducibility
    base_seed = 42 
    
    # Track overall metrics across all trials
    overall_prompt_tokens = 0
    overall_completion_tokens = 0
    overall_recalls = []
    overall_precisions = []
    overall_f1s = []
    overall_execution_times = [] 

    print(f"\n==================================================")
    print(f"--- Naive Join Run: {num_trials} Trials of {sample_size}x{sample_size} Grid (NO Chain-of-Thought) ---")
    
    script_start_time = time.time()
    
    for trial in range(1, num_trials + 1):
        print(f"\n>>> STARTING TRIAL {trial} <<<")
        
        # NEW: Set random_state using the base seed and the trial number
        # We use a different offset for df_a and df_b to ensure they don't accidentally mirror each other's sampling patterns
        df_a = full_df_a.sample(n=sample_size, random_state=base_seed + trial)
        df_b = full_df_b.sample(n=sample_size, random_state=base_seed + trial + 100)

        trial_prompt_tokens = 0
        trial_completion_tokens = 0

        # 2. Establish Ground Truth for this trial
        expected_matches = []
        for i, row_a in df_a.iterrows():
            for j, row_b in df_b.iterrows():
                if row_a['sentiment'] == row_b['sentiment']:
                    expected_matches.append((i, j))
        
        print(f"Ground Truth: Found {len(expected_matches)} matching pairs out of {sample_size * sample_size} possible.")
        print("Calling LLM for semantic matching (Pair-by-Pair)...")

        llm_matches = []
        
        start_time = time.time()
        
        for i, row_a in df_a.iterrows():
            for j, row_b in df_b.iterrows():
                prompt = f"""
                Task: Compare the sentiment of two movie reviews.
                
                Review A: "{row_a['review'][:400]}"
                Review B: "{row_b['review'][:400]}"
                
                Do they have the SAME sentiment (both positive or both negative)?
                Answer only YES or NO.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2, 
                    temperature=0,
                    seed=42
                )

                trial_prompt_tokens += response.usage.prompt_tokens
                trial_completion_tokens += response.usage.completion_tokens
                
                output = response.choices[0].message.content.strip().upper()
                if "YES" in output:
                    llm_matches.append((i, j))

        elapsed_time = time.time() - start_time
        overall_execution_times.append(elapsed_time) 

        # 3. Calculate Confusion Matrix Elements
        expected_set = set(expected_matches)
        llm_set = set(llm_matches)

        true_positives = len(expected_set.intersection(llm_set))
        false_positives = len(llm_set - expected_set)
        false_negatives = len(expected_set - llm_set)

        # 4. Calculate Precision, Recall, and F1 Score
        recall = (true_positives / len(expected_set)) * 100 if expected_set else 100.0
        precision = (true_positives / len(llm_set)) * 100 if llm_set else 100.0
        
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        overall_recalls.append(recall)
        overall_precisions.append(precision)
        overall_f1s.append(f1_score)
        
        print(f"--- Trial {trial} Results ---")
        print(f"Execution Time: {elapsed_time:.2f} seconds")
        print(f"Matches Predicted by LLM: {len(llm_matches)}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives (Hallucinations): {false_positives}")
        print(f"False Negatives (Misses): {false_negatives}")
        print(f"Recall:    {recall:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"F1 Score:  {f1_score:.2f}%")

        print(f"--- Trial {trial} Token Usage ---")
        print(f"Input (Prompt) Tokens: {trial_prompt_tokens}")
        print(f"Output (Completion) Tokens: {trial_completion_tokens}")
        print(f"Total Tokens Used: {trial_prompt_tokens + trial_completion_tokens}")
        
        overall_prompt_tokens += trial_prompt_tokens
        overall_completion_tokens += trial_completion_tokens

    # 5. Output Overall Summary
    avg_recall = sum(overall_recalls) / len(overall_recalls)
    avg_precision = sum(overall_precisions) / len(overall_precisions)
    avg_f1 = sum(overall_f1s) / len(overall_f1s)
    
    total_llm_time = sum(overall_execution_times)
    avg_llm_time = total_llm_time / len(overall_execution_times)
    total_script_time = time.time() - script_start_time
    
    # Calculate per-pair averages
    total_comparisons = num_trials * (sample_size * sample_size)
    grand_total_tokens = overall_prompt_tokens + overall_completion_tokens
    avg_tokens_per_pair = grand_total_tokens / total_comparisons
    avg_time_per_pair = total_llm_time / total_comparisons

    print(f"\n==================================================")
    print(f"--- OVERALL SUMMARY ACROSS {num_trials} TRIALS ---")
    print(f"Average Recall:    {avg_recall:.2f}%")
    print(f"Average Precision: {avg_precision:.2f}%")
    print(f"Average F1 Score:  {avg_f1:.2f}%")
    print(f"--- Token Usage ---")
    print(f"Average Tokens per Comparison:   {avg_tokens_per_pair:.2f}")
    print(f"Grand Total Tokens Used:         {grand_total_tokens}")
    print(f"--- Execution Time ---")
    print(f"Average Time per Comparison:     {avg_time_per_pair:.2f} seconds")
    print(f"Total Time in LLM Calls:         {total_llm_time:.2f} seconds")
    print(f"Total Script Execution Time:     {total_script_time:.2f} seconds")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_naive_metrics_no_cot()