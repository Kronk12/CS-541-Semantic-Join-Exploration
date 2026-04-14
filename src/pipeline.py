import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def serialize_row(row, columns):
  """Converts a dataframe row into a structured string context."""
  return " | ".join([f"{col}: {row[col]}" for col in columns])

def get_cluster_samples(df, cluster_id, serialized_col, n_samples=5):
    """Extracts sample serialized rows from a cluster."""
    cluster_rows = df[df['cluster'] == cluster_id]
    n = min(n_samples, len(cluster_rows))
    samples = cluster_rows.sample(n=n, random_state=42)[serialized_col].tolist()
    return "\n".join([f"- {sample}" for sample in samples])


class SemanticJoiner:
  def __init__(self, model_name="gpt-4o-mini", embed_model='distilbert-base-uncased-finetuned-sst-2-english'):
    print("Initializing SemanticJoiner...")
    self.model_name = model_name
    self.total_tokens = 0

    print("Initializing embedding model...")
    self.embedder = SentenceTransformer(embed_model)


  def step1_cluster(self, table_a, table_b, num_clusters=2):
    print(f"\n--- STEP 1: Clustering (k={num_clusters}) ---")
    total_start_time = time.time()

    # Serialize rows
    cols_a = list(table_a)
    cols_b = list(table_b)
    table_a['_serialized'] = table_a.apply(lambda r: serialize_row(r, cols_a), axis=1)
    table_b['_serialized'] = table_b.apply(lambda r: serialize_row(r, cols_b), axis=1)
    # print(table_a)
    # print(table_b)

    # Generate embeddings for both tables
    print("Generating embeddings for serialized rows in Table A and Table B...")
    embed_start_time = time.time()
    emb_a = self.embedder.encode(table_a['_serialized'].tolist())
    emb_b = self.embedder.encode(table_b['_serialized'].tolist())
    embed_time = time.time() - embed_start_time
    print(f"  -> Embeddings generated in {embed_time:.2f} seconds.")

    # Cluster
    print("Running K-Means algorithm...")
    kmeans_start_time = time.time()
    table_a['cluster'] = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(emb_a)
    table_b['cluster'] = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(emb_b)
    kmeans_time = time.time() - kmeans_start_time
    print(f"  -> Clustering finished in {kmeans_time:.2f} seconds.")

    total_time = time.time() - total_start_time

    print(f"\n--- Clustering Complete! ---")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Table A clusters: {table_a['cluster'].value_counts().to_dict()}")
    print(f"Table B clusters: {table_b['cluster'].value_counts().to_dict()}")

    # Preview a cluster
    print("\nPreviewing Table A, Cluster 0 (First 2 entries):")
    print(table_a[table_a['cluster'] == 0]['_serialized'].str[:100].tolist()[:2])

    return table_a, table_b


  def step2_filter_clusters(self, table_a, table_b, join_predicate):
    print("\n--- STEP 2: LLM Cluster Filtering ---")

    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    valid_cluster_pairs = []
  
    clusters_a = table_a['cluster'].unique()
    clusters_b = table_b['cluster'].unique()

    total_checks = len(clusters_a) * len(clusters_b)
    print(f"Evaluating {total_checks} cluster combinations against predicate...")

    for ca in clusters_a:
      samples_a = get_cluster_samples(table_a, ca, '_serialized')
      for cb in clusters_b:
        samples_b = get_cluster_samples(table_b, cb, '_serialized')

        prompt = f"""
          You are evaluating two datasets for a database join.
                
          JOIN PREDICATE: "{join_predicate}"
                
          Table A samples:
          {samples_a}
                
          Table B samples:
          {samples_b}
                
          Based on the samples, is it logically possible that ANY row in Table A could match ANY row in Table B according to the join predicate? 
          If they represent entirely incompatible concepts based on the predicate, return false. If there is even a slight chance of a match, return true.
                
          Return a JSON object exactly matching this structure:
          {{
            "match_possible": true or false
          }}
          """
                
        response = client.chat.completions.create(
          model=self.model_name,
          messages=[{"role": "user", "content": prompt}],
          response_format={"type": "json_object"},
          temperature=0
        )

        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens

        result = json.loads(response.choices[0].message.content)
        if result.get("match_possible", False):
          valid_cluster_pairs.append((ca, cb))
          print(f"  [KEEP] Pair (A:{ca}, B:{cb})")
        else:
          print(f"  [DROP] Pair (A:{ca}, B:{cb})")

    elapsed_time = time.time() - start_time
    self.total_tokens += total_prompt_tokens + total_completion_tokens

    print(f"Kept {len(valid_cluster_pairs)}/{total_checks} cluster pairs.")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Prompt Tokens Used: {total_prompt_tokens}")
    print(f"Completion Tokens Used: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")
    return valid_cluster_pairs


  def step3_join(self, table_a, table_b, cluster_pairs, join_predicate):
    print("\n--- STEP 3: LLM Semantic Join ---")

    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    all_matches = []
    print(f"Processing {len(cluster_pairs)} cluster pairs...")
    for ca, cb in cluster_pairs:
      rows_a = table_a[table_a['cluster'] == ca]
      rows_b = table_b[table_b['cluster'] == cb]

      # Format block text using dataframe indices as unique IDs
      block_a_text = "\n".join([f"ID A-{idx}: {row['_serialized'][:500]}" for idx, row in rows_a.iterrows()])
      block_b_text = "\n".join([f"ID B-{idx}: {row['_serialized'][:500]}" for idx, row in rows_b.iterrows()])

      prompt = f"""
        You are a high-precision data joiner. Your task is to find all pairs of records that satisfy the following JOIN PREDICATE.
            
        JOIN PREDICATE: "{join_predicate}"
            
        Compare EVERY item in TABLE A against EVERY item in TABLE B. 
            
        TABLE A:
        {block_a_text}
            
        TABLE B:
        {block_b_text}
            
        Return ONLY a JSON dictionary where each key is a Table A ID, and its value is a list of all matching Table B IDs. 
        If a Table A ID has no matches, return an empty list for that key.
            
        Example format:
        {{
          "matches": {{
            "A-0": ["B-1", "B-5"],
            "A-1": []
          }}
        }}
        """
      
      try:
        response = client.chat.completions.create(
          model=self.model_name,
          messages=[{"role": "user", "content": prompt}],
          response_format={"type": "json_object"},
          temperature=0
        )
        
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens

        result = json.loads(response.choices[0].message.content)
        matches_dict = result.get("matches", {})
        
        pair_match_count = 0
        for a_id, b_list in matches_dict.items():
          # Strip 'A-' and 'B-' to get raw indices
          raw_a_id = a_id.replace("A-", "")
          for b_id in b_list:
              raw_b_id = b_id.replace("B-", "")
              all_matches.append((raw_a_id, raw_b_id))
              pair_match_count += 1
                  
        print(f"  Found {pair_match_count} matches in Pair (A:{ca}, B:{cb})")
   
      except Exception as e:
          print(f"  Error processing pair (A:{ca}, B:{cb}): {e}")

    elapsed_time = time.time() - start_time
    self.total_tokens += total_prompt_tokens + total_completion_tokens

    print(f"\nJoin Complete! Total matches found: {len(all_matches)}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Prompt Tokens Used: {total_prompt_tokens}")
    print(f"Completion Tokens Used: {total_completion_tokens}")
    print(f"Total Tokens Used: {total_prompt_tokens + total_completion_tokens}")

    return pd.DataFrame(all_matches, columns=['Table_A_Index', 'Table_B_Index'])
  

  def print_summary(self):
    print(f"Total LLM Tokens Used across pipeline: {self.total_tokens}")


def main():
  # Load your tables
  # table_a = pd.read_csv('data/table_a.csv')
  # table_b = pd.read_csv('data/table_b.csv')

  # Mock Data
  table_a = pd.DataFrame({
    'company': ['TechCorp', 'GreenEnergy', 'HealthPlus'],
    'description': ['Builds software for servers', 'Installs solar panels', 'Makes medical devices']
  })
    
  table_b = pd.DataFrame({
    'job_seeker': ['Alice', 'Bob', 'Charlie'],
    'resume': ['Software engineer with 5 years backend experience.', 'Registered nurse.', 'Expert in renewable energy grids.']
  })

  predicate = "Join if the job seeker's resume skills directly align with the company's description/industry."

  joiner = SemanticJoiner()
  df_a, df_b = joiner.step1_cluster(table_a, table_b)
  cluster_pairs = joiner.step2_filter_clusters(df_a, df_b, predicate)
  joined_indices_df = joiner.step3_join(df_a, df_b, cluster_pairs, predicate)
  
  print(joined_indices_df)


if __name__ == "__main__":
  main()