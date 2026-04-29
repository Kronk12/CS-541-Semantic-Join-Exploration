import pandas as pd
import os

def preprocess_magellan_data():
    print("Loading raw Magellan datasets...")
    # Magellan datasets require latin1 encoding
    amazon_df = pd.read_csv('data/Amazon.csv', encoding='latin1')
    google_df = pd.read_csv('data/GoogleProducts.csv', encoding='latin1')
    mapping_df = pd.read_csv('data/Amzon_GoogleProducts_perfectMapping.csv', encoding='latin1')
    
    # 1. Sample exactly 100 matching pairs (True Positives)
    # Using random_state=42 ensures you get the exact same 100 rows every time you run this
    subset_mapping = mapping_df.sample(n=100, random_state=42).reset_index(drop=True)
    
    amazon_ids = subset_mapping['idAmazon'].tolist()
    google_ids = subset_mapping['idGoogleBase'].tolist()
    
    # 2. Extract the full records for these specific IDs
    amazon_full = amazon_df[amazon_df['id'].isin(amazon_ids)].drop_duplicates(subset=['id'])
    google_full = google_df[google_df['id'].isin(google_ids)].drop_duplicates(subset=['id'])
    
    print(f"Extracted {len(amazon_full)} Amazon rows and {len(google_full)} Google rows.")

    # 3. Create the slim versions (ID + Title/Name only)
    amazon_slim = amazon_full[['id', 'title']]
    google_slim = google_full[['id', 'name']]
    
    # 4. Save everything to the data directory
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    amazon_full.to_csv(f'{output_dir}/amazon_100_full.csv', index=False)
    amazon_slim.to_csv(f'{output_dir}/amazon_100_slim.csv', index=False)
    
    google_full.to_csv(f'{output_dir}/google_100_full.csv', index=False)
    google_slim.to_csv(f'{output_dir}/google_100_slim.csv', index=False)
    
    # Save the isolated ground truth for these 100 pairs
    subset_mapping.to_csv(f'{output_dir}/magellan_100_ground_truth.csv', index=False)
    
    print(f"\nSuccessfully generated 4 tables and 1 mapping file in the '{output_dir}/' directory.")

if __name__ == "__main__":
    preprocess_magellan_data()