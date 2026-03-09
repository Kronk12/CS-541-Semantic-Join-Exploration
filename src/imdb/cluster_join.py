import pandas as pd
from openai import OpenAI
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_cluster_join(model_name, pairs_file, output_file):
    print(f"\n==================================================")
    mode_name = "FILTERED CLUSTERS" if pairs_file else "ALL CLUSTERS (No Filter)"
    print(f"Executing Cluster Join | Mode: {mode_name} | Model: {model_name}")
    print(f"==================================================")
    
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Load the N=50 clustered data
    table_a = pd.read_csv('data/table_a_clustered_distilbert.csv')
    table_b = pd.read_csv('data/table_b_clustered_distilbert.csv')
    
    # Load the specific survivor pairs OR generate all pairs
    if pairs_file:
        if not os.path.exists(pairs_file):
            print(f"Error: {pairs_file} not found.")
            return
        with open(pairs_file, 'r') as f:
            valid_pairs = json.load(f)
    else:
        clusters_a = table_a['cluster'].unique()
        clusters_b = table_b['cluster'].unique()
        # Create a cross-product of all clusters
        valid_pairs = [(int(ca), int(cb)) for ca in clusters_a for cb in clusters_b]

    all_matches = []
    print(f"Processing {len(valid_pairs)} cluster pairs...")

    for ca, cb in valid_pairs:
        rows_a = table_a[table_a['cluster'] == ca]
        rows_b = table_b[table_b['cluster'] == cb]
        
        # Format block text using global index
        block_a_text = "\n".join([f"ID A-{idx}: {row['review'][:400]}" for idx, row in rows_a.iterrows()])
        block_b_text = "\n".join([f"ID B-{idx}: {row['review'][:400]}" for idx, row in rows_b.iterrows()])

        # The proven Adjacency List Prompt
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

        # Dynamically build kwargs
        api_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": { "type": "json_object" }
        }
        
        if "gpt-5-mini" not in model_name:
            api_kwargs["temperature"] = 0 

        try:
            response = client.chat.completions.create(**api_kwargs)
            
            # Track tokens
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
            # Parse result (Adjacency List unpacking)
            result = json.loads(response.choices[0].message.content)
            matches_dict = result.get("matches", {})
            
            pair_match_count = 0
            for a_id, b_list in matches_dict.items():
                for b_id in b_list:
                    all_matches.append((a_id, b_id))
                    pair_match_count += 1
                    
            print(f"  Found {pair_match_count} matches in Pair ({ca}, {cb})")
            
        except Exception as e:
            print(f"  Error processing pair ({ca}, {cb}): {e}")

    elapsed_time = time.time() - start_time
    
    print(f"\nJoin Complete! Total Semantic Matches Found: {len(all_matches)}")
    
    if all_matches:
        pd.DataFrame(all_matches, columns=['Table_A_ID', 'Table_B_ID']).to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    else:
        print("No matches to save.")

    # Output Metrics
    print(f"\n--- Metrics for {model_name} on {pairs_file if pairs_file else 'ALL CLUSTERS'} ---")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Total Input Tokens: {total_prompt_tokens}")
    print(f"Total Output Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"==================================================\n")


if __name__ == "__main__":
    # 1. Run with Filtered Clusters (The 12 surviving pairs)
    run_cluster_join(
        model_name="gpt-4o",
        pairs_file="data/valid_cluster_pairs_distilbert.json",
        output_file="data/final_matches_filtered_clusters.csv"
    )
    
    # 2. Run with All Clusters (All 25 pairs, effectively just blocking by topic)
    run_cluster_join(
        model_name="gpt-4o",
        pairs_file=None,
        output_file="data/final_matches_all_clusters.csv"
    )