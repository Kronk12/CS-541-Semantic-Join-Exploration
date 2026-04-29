import pandas as pd
from openai import OpenAI
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Setup LLM Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_cluster_samples(df, cluster_id):
    """Get up to 3 sample reviews (or the whole cluster if < 3) from a specific cluster."""
    cluster_rows = df[df['cluster'] == cluster_id]
    n_samples = min(3, len(cluster_rows))
    
    samples = cluster_rows.sample(n=n_samples, random_state=42)
    
    sample_texts = []
    for idx, text in enumerate(samples['review']):
        # Truncate each sample to 300 chars to keep token usage manageable
        sample_texts.append(f"Sample {idx+1}: \"{text[:300]}...\"")
        
    return "\n".join(sample_texts)

def run_filter(model_name, temperature_val, output_file):
    # 1. Initialize token trackers and timer
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()

    table_a = pd.read_csv('data/table_a_clustered_100.csv')
    table_b = pd.read_csv('data/table_b_clustered_100.csv')
    
    clusters_a = table_a['cluster'].unique()
    clusters_b = table_b['cluster'].unique()
    
    valid_pairs = []
    total_checks = len(clusters_a) * len(clusters_b)
    
    print(f"Beginning Cluster-Level Filtering ({total_checks} possible pairs)...")
    print(f"Model: {model_name} | Target Temperature: {temperature_val}")
    print(f"Sampling up to 3 reviews per cluster to represent sentiment.")

    for ca in clusters_a:
        samples_a = get_cluster_samples(table_a, ca)
        
        for cb in clusters_b:
            samples_b = get_cluster_samples(table_b, cb)
            
            # THE UPDATED PROMPT: Strict rules for high-recall filtering
            prompt = f"""
            You are a highly conservative data filtering assistant. We are performing a semantic join on movie reviews.
            The join condition is: "The two reviews have the same sentiment (both positive or both negative)."
            
            Review Samples from Cluster A:
            {samples_a}
            
            Review Samples from Cluster B:
            {samples_b}
            
            Based ONLY on these samples, evaluate if it is possible that movies in Cluster A and Cluster B share the same sentiment.
            
            STRICT RULES:
            1. DEFAULT TO MATCH: If there is any uncertainty, mixed sentiment, neutral tone, or even a slight possibility of overlap, you MUST respond "YES".
            2. HIGH THRESHOLD FOR EXCLUSION: Only respond "NO" if the clusters are CLEARLY, UNAMBIGUOUSLY, and ENTIRELY contradictory in sentiment (e.g., Cluster A is 100% overwhelmingly positive and Cluster B is 100% overwhelmingly negative).
            
            Respond with a single word: YES or NO.
            """
            
            # Dynamically build the API arguments to avoid the gpt-5-mini temperature crash
            api_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Only add temperature if it's NOT a reasoning model like gpt-5-mini
            if "gpt-5-mini" not in model_name:
                api_kwargs["temperature"] = temperature_val
                api_kwargs["max_tokens"] = 2

            response = client.chat.completions.create(**api_kwargs)
            
            # 2. Track tokens
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
            answer = response.choices[0].message.content.strip().upper()
            
            # Cast to native python types to ensure clean JSON overwriting
            ca_val = ca.item() if hasattr(ca, 'item') else ca
            cb_val = cb.item() if hasattr(cb, 'item') else cb

            if "YES" in answer:
                valid_pairs.append((ca_val, cb_val))
                print(f"  [KEEP] Cluster A:{ca_val} and Cluster B:{cb_val}")
            else:
                print(f"  [DROP] Cluster A:{ca_val} and Cluster B:{cb_val}")

    # 3. Stop timer and save the valid pairings
    elapsed_time = time.time() - start_time
    with open(output_file, 'w') as f:
        json.dump(valid_pairs, f)
    
    # 4. Output Final Metrics
    print(f"\n--- {model_name} Filtering Complete ---")
    print(f"Kept {len(valid_pairs)}/{total_checks} cluster pairs.")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Total Input (Prompt) Tokens: {total_prompt_tokens}")
    print(f"Total Output (Completion) Tokens: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"Results saved to: {output_file}")
    print(f"==================================================\n")

if __name__ == "__main__":
    print("==================================================")
    print("RUN 1: Standard Frontier Model (gpt-4o)")
    print("==================================================")
    run_filter(
        model_name="gpt-4o", 
        temperature_val=0, 
        output_file='data/valid_cluster_pairs_gpt4o_100.json'
    )

    # print("==================================================")
    # print("RUN 2: Fast/Efficient Model (gpt-5-mini)")
    # print("==================================================")
    # run_filter(
    #     model_name="gpt-5-mini", 
    #     temperature_val=0, # This will now be safely ignored by the script
    #     output_file='data/valid_cluster_pairs_gpt5_conservative.json'
    # )