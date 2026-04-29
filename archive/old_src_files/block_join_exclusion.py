import pandas as pd
from openai import OpenAI
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_block_join_exclusion(model_name, pairs_file, output_file):
    print(f"\n==================================================")
    print(f"Executing Block Join (EXCLUSION METHOD) | Model: {model_name}")
    print(f"Using Filter File: {pairs_file}")
    print(f"==================================================")
    
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Load data
    table_a = pd.read_csv('data/table_a_clustered_100.csv')
    table_b = pd.read_csv('data/table_b_clustered_100.csv')
    
    if not os.path.exists(pairs_file):
        print(f"Error: {pairs_file} not found. Did the filtering script complete?")
        return
        
    with open(pairs_file, 'r') as f:
        valid_pairs = json.load(f)

    all_matches_formatted = []
    print(f"Processing {len(valid_pairs)} surviving cluster pairs...")

    for ca, cb in valid_pairs:
        rows_a = table_a[table_a['cluster'] == ca]
        rows_b = table_b[table_b['cluster'] == cb]
        
        # 1. Establish all possible combinations for this block pair
        all_possible_pairs = set()
        for idx_a, row_a in rows_a.iterrows():
            for idx_b, row_b in rows_b.iterrows():
                all_possible_pairs.add((idx_a, idx_b))

        # 2. Format the text for the prompt
        block_a_text = "\n".join([f"ID A-{idx}: {row['review'][:200]}" for idx, row in rows_a.iterrows()])
        block_b_text = "\n".join([f"ID B-{idx}: {row['review'][:200]}" for idx, row in rows_b.iterrows()])

        # 3. The Exclusion Prompt
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
        # prompt = f"""
        #         Compare EVERY item in TABLE A against EVERY item in TABLE B. 
        #         Identify all pairs of reviews that have the SAME sentiment (both positive or both negative).
                
        #         TABLE A:
        #         {block_a_text}
                
        #         TABLE B:
        #         {block_b_text}
                
        #         Return a JSON object with a single key "matches" containing a list of ID pairs that MATCH in sentiment.
        #         Example: {{"matches": [["A-45", "B-12"], ["A-47", "B-15"]]}}
        #         If there are no matches, return {{"matches": []}}.
        #         """

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0
            )
            
            # Track tokens
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
            # 4. Parse non-matches
            llm_non_matches = set()
            result = json.loads(response.choices[0].message.content)
            
            for pair in result.get("non_matches", []):
                id_a = int(pair[0].replace('A-', ''))
                id_b = int(pair[1].replace('B-', ''))
                llm_non_matches.add((id_a, id_b))
                
            # 5. Invert logic: Matches = All Pairs - Non-Matches
            llm_matches = all_possible_pairs - llm_non_matches
            
            # Format back to "A-X", "B-X" for saving
            for match in llm_matches:
                all_matches_formatted.append((f"A-{match[0]}", f"B-{match[1]}"))
                
            print(f"  Pair ({ca}, {cb}): {len(all_possible_pairs)} combinations -> LLM excluded {len(llm_non_matches)} -> {len(llm_matches)} inferred matches.")
            
        except Exception as e:
            print(f"  Error processing pair ({ca}, {cb}): {e}")

    elapsed_time = time.time() - start_time
    
    print(f"\nJoin Complete! Total Semantic Matches Inferred: {len(all_matches_formatted)}")
    
    if all_matches_formatted:
        pd.DataFrame(all_matches_formatted, columns=['Table_A_ID', 'Table_B_ID']).to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    else:
        print("No matches to save.")

    # Output Metrics
    print(f"\n--- Metrics for {model_name} (Exclusion Method) ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Total Input Tokens: {total_prompt_tokens}")
    print(f"Total Output Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Run the winning configuration: GPT-4o Block Join on GPT-5-mini filtered clusters
    run_block_join_exclusion(
        model_name="gpt-4o",
        pairs_file="data/valid_cluster_pairs_gpt4o_100.json",
        output_file="data/final_matches_4o_100_exclusion.csv"
    )