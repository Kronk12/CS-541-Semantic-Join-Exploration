import pandas as pd
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_arbitrary_blocks(block_size=10, prompt_type="inclusion"):
    print(f"\n==================================================")
    print(f"Executing Arbitrary Block Join | Size: {block_size}x{block_size} | Prompt: {prompt_type.upper()}")
    print(f"==================================================")
    
    # Load the EXACT SAME 100x100 dataset
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')
    
    # Establish local ground truth
    ground_truth = set()
    all_possible = set()
    for i, row_a in table_a.iterrows():
        for j, row_b in table_b.iterrows():
            all_possible.add((i, j))
            if row_a['sentiment'] == row_b['sentiment']:
                ground_truth.add((i, j))
                
    total_expected = len(ground_truth)
    print(f"Absolute Ground Truth: {total_expected} total matching pairs out of 10,000.")

    # Chunk the dataframes into lists of dataframes (blocks)
    blocks_a = [table_a.iloc[i:i + block_size] for i in range(0, len(table_a), block_size)]
    blocks_b = [table_b.iloc[i:i + block_size] for i in range(0, len(table_b), block_size)]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()
    
    all_llm_matches = set()
    total_loops = len(blocks_a) * len(blocks_b)
    current_loop = 1

    # Cross-join the blocks
    for i, b_a in enumerate(blocks_a):
        for j, b_b in enumerate(blocks_b):
            print(f"Processing Block Pair {current_loop}/{total_loops}...", end="\r")
            current_loop += 1
            
            # 1. Establish block-level combinations
            block_pairs = set()
            for idx_a, _ in b_a.iterrows():
                for idx_b, _ in b_b.iterrows():
                    block_pairs.add((idx_a, idx_b))
            
            # 2. Format Text
            block_a_text = "\n".join([f"ID A-{idx}: {row['review'][:400]}" for idx, row in b_a.iterrows()])
            block_b_text = "\n".join([f"ID B-{idx}: {row['review'][:400]}" for idx, row in b_b.iterrows()])

            # 3. Dynamic Prompting
            if prompt_type == "exclusion":
                prompt = f"""
        Find all pairs (Review A, Review B) that have DIFFERENT sentiments (One is Positive and the other is Negative).
        
        TABLE A:
        {block_a_text}
        
        TABLE B:
        {block_b_text}
        
        Return ONLY a JSON object with the key "non_matches".
        Example: {{"non_matches": [["A-ID", "B-ID"], ...]}}
        If all pairs have the same sentiment (0 non-matches), return {{"non_matches": []}}.
        """
            else:
                # The New Adjacency List Prompt
                prompt = f"""
        You are a high-precision data joiner. Your task is to find all pairs of movie reviews that have the SAME sentiment (both positive or both negative).
        
        Compare EVERY item in TABLE A against EVERY item in TABLE B. 
        
        TABLE A:
        {block_a_text}
        
        TABLE B:
        {block_b_text}
        
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

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" },
                    temperature=0
                )
                
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
                
                # 4. Dynamic Parsing
                result = json.loads(response.choices[0].message.content)
                
                if prompt_type == "exclusion":
                    llm_non_matches = set()
                    for pair in result.get("non_matches", []):
                        id_a = int(pair[0].replace('A-', ''))
                        id_b = int(pair[1].replace('B-', ''))
                        llm_non_matches.add((id_a, id_b))
                    # Invert logic to get matches
                    llm_matches = block_pairs - llm_non_matches
                else:
                    # New Adjacency List Parsing
                    llm_matches = set()
                    matches_dict = result.get("matches", {})
                    for a_id, b_list in matches_dict.items():
                        try:
                            clean_a = int(a_id.replace('A-', ''))
                            for b_id in b_list:
                                clean_b = int(b_id.replace('B-', ''))
                                llm_matches.add((clean_a, clean_b))
                        except ValueError:
                            pass # Safely ignore formatting hallucinations
                        
                all_llm_matches.update(llm_matches)
                
            except Exception as e:
                print(f"\n  Error processing block pair {current_loop-1}: {e}")

    elapsed_time = time.time() - start_time

    # Evaluate final results
    true_positives = len(ground_truth.intersection(all_llm_matches))
    false_positives = len(all_llm_matches - ground_truth)
    false_negatives = len(ground_truth - all_llm_matches)

    recall = (true_positives / total_expected) * 100 if total_expected > 0 else 0
    precision = (true_positives / len(all_llm_matches)) * 100 if len(all_llm_matches) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n\n--- Metrics for {prompt_type.upper()} Prompt ({block_size}x{block_size}) ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Recall: {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")
    print(f"Total Input Tokens: {total_prompt_tokens}")
    print(f"Total Output Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Test 1: The Adjacency List Inclusion prompt
    run_arbitrary_blocks(block_size=10, prompt_type="inclusion")
    
    # # Test 2: The standard Exclusion prompt
    # run_arbitrary_blocks(block_size=10, prompt_type="exclusion")