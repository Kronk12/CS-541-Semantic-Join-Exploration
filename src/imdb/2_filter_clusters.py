import pandas as pd
from openai import OpenAI
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_cluster_samples(df, cluster_id):
    """Get up to 10 sample reviews from a specific cluster."""
    cluster_rows = df[df['cluster'] == cluster_id]
    n_samples = min(10, len(cluster_rows))
    
    samples = cluster_rows.sample(n=n_samples, random_state=42)
    
    sample_texts = []
    for idx, text in enumerate(samples['review']):
        sample_texts.append(f"Sample {idx+1}: \"{text[:300]}...\"")
        
    return "\n".join(sample_texts)

def profile_cluster_sentiment(model_name, temperature_val, samples):
    """Asks the LLM to classify a single cluster."""
    prompt = f"""
    Determine the DOMINANT sentiment (majority rule) of these movie reviews.
    
    Reviews:
    {samples}
    
    Return a JSON object exactly matching this structure:
    {{
        "dominant_sentiment": "positive or negative"
    }}
    """
    
    api_kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": { "type": "json_object" }
    }
    
    if "gpt-5-mini" not in model_name:
        api_kwargs["temperature"] = temperature_val

    response = client.chat.completions.create(**api_kwargs)
    
    result = json.loads(response.choices[0].message.content)
    sentiment = result.get("dominant_sentiment", "unknown").lower()
    
    return sentiment, response.usage.prompt_tokens, response.usage.completion_tokens

def run_filter(model_name, temperature_val, output_file):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    table_a = pd.read_csv('data/table_a_clustered_distilbert.csv')
    table_b = pd.read_csv('data/table_b_clustered_distilbert.csv')
    
    clusters_a = table_a['cluster'].unique()
    clusters_b = table_b['cluster'].unique()
    
    print(f"Profiling {len(clusters_a)} clusters from Table A and {len(clusters_b)} clusters from Table B...")
    
    # 1. Profile Table A
    sentiments_a = {}
    for ca in clusters_a:
        samples = get_cluster_samples(table_a, ca)
        sent, p_tok, c_tok = profile_cluster_sentiment(model_name, temperature_val, samples)
        sentiments_a[ca.item() if hasattr(ca, 'item') else ca] = sent
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        print(f"  Table A, Cluster {ca}: {sent}")

    # 2. Profile Table B
    sentiments_b = {}
    for cb in clusters_b:
        samples = get_cluster_samples(table_b, cb)
        sent, p_tok, c_tok = profile_cluster_sentiment(model_name, temperature_val, samples)
        sentiments_b[cb.item() if hasattr(cb, 'item') else cb] = sent
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        print(f"  Table B, Cluster {cb}: {sent}")

    # 3. Match Locally in Python
    valid_pairs = []
    total_checks = len(clusters_a) * len(clusters_b)
    
    print(f"\nMatching pairs logically...")
    for ca, sent_a in sentiments_a.items():
        for cb, sent_b in sentiments_b.items():
            if sent_a == sent_b and sent_a in ['positive', 'negative']:
                valid_pairs.append((ca, cb))
                print(f"  [KEEP] A:{ca} and B:{cb} (Both {sent_a})")
            else:
                print(f"  [DROP] A:{ca} ({sent_a}) and B:{cb} ({sent_b})")

    # 4. Save and Output
    elapsed_time = time.time() - start_time
    with open(output_file, 'w') as f:
        json.dump(valid_pairs, f)
    
    print(f"\n--- {model_name} Filtering Complete ---")
    print(f"Kept {len(valid_pairs)}/{total_checks} cluster pairs.")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    print(f"Results saved to: {output_file}")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_filter(
        model_name="gpt-4o", 
        temperature_val=0, 
        output_file='data/valid_cluster_pairs_distilbert.json'
    )