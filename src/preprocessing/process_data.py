import pandas as pd
import requests
import io

def setup_imdb_data():
    # URL to a clean CSV version of the IMDb 50k dataset
    df = pd.read_csv("/Users/Total/Documents/GitHub/CS-541-Semantic-Join-Exploration/data/IMDB Dataset.csv")
    
    # 1. Take a small sample to stay within your API/Time budget (100 rows)
    # This matches the sample size used in the 'Reviews' benchmark [cite: 2798]
    sample_df = df.sample(n=200, random_state=42).reset_index(drop=True)
    
    # 2. Split into two tables to simulate a Join
    table_a = sample_df.iloc[:100].copy()
    table_b = sample_df.iloc[100:].copy()
    
    # 3. Save to your /data folder
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
        
    table_a.to_csv('data/table_a_100.csv', index=False)
    table_b.to_csv('data/table_b_100.csv', index=False)
    
    print(f"Successfully created data/table_a_100.csv ({len(table_a)} rows)")
    print(f"Successfully created data/table_b_100.csv ({len(table_b)} rows)")
    print("\nSample Review from Table A:")
    print(table_a['review'].iloc[0][:150] + "...")

if __name__ == "__main__":
    setup_imdb_data()