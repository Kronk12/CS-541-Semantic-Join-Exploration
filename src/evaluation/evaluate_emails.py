import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

import pandas as pd
from semantic_join import semantic_join
from simulate import export_simulation_data

# Load your dataset
df_a = pd.read_csv('data/table_a_emails.csv')
df_b = pd.read_csv('data/table_b_emails.csv')

# Define the target ratios for the grid search
target_ratios = [0.1, 0.2, 0.05]

for ratio in target_ratios:
    # Convert decimal percentage to integer divisor
    divisor = int(1 / ratio)
    
    print(f"\n" + "="*60)
    print(f"Running master evaluation for ratio: {ratio} (Divisor: {divisor})")
    print("="*60)
    
    # Run the join with threshold=-1.0 to get the master evaluation
    result = semantic_join(
        table_a=df_a,
        table_b=df_b,
        predicate="the two texts contradict each other",
        schema_a=["email"],
        schema_b=["email"],
        filter_threshold=-1.0,
        cluster_size_limit=-1,
        force_strategy="pairwise",
        embedding="all-mpnet-base-v2",
        clustering="kmeans",
        block_size=15,
        cluster_ratio=divisor, 
        verbose=False # Set to False to keep the console output clean during the loop
    )

    # Dynamically name the output file based on the ratio
    output_filename = f"logs/emails_master_log_ratio_{ratio}.json"

    # Export everything, including up to 3 sample rows per cluster
    export_simulation_data(
        result=result, 
        table_a=df_a, 
        table_b=df_b, 
        filepath=output_filename,
        num_samples=3
    )