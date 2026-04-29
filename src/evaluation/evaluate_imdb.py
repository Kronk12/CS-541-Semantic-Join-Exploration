import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

from semantic_join import semantic_join
from simulate import export_simulation_data

def evaluate_imdb_pairwise():
    # Load the IMDB datasets
    df_a = pd.read_csv('data/table_a.csv')
    df_b = pd.read_csv('data/table_b.csv')

    # Define the target ratios for the grid search
    target_ratios = [0.025, 0.05, 0.075, 0.1]

    # Test both with and without projection
    projection_states = [False, True]

    for ratio in target_ratios:
        divisor = int(1 / ratio)
        
        for force_proj in projection_states:
            print("\n" + "="*60)
            print(f"Running IMDB Pairwise Eval | Ratio: {ratio} (Divisor: {divisor}) | Projection: {force_proj}")
            print("="*60)
            
            result = semantic_join(
                table_a=df_a,
                table_b=df_b,
                predicate="Both reviews express the same sentiment (Positive or Negative)",
                schema_a=["review"],
                schema_b=["review"],
                
                # --- FORCING PARAMETERS ---
                force_strategy="pairwise",
                force_projection=force_proj,
                embedding="distilbert-base-uncased-finetuned-sst-2-english",
                clustering="kmeans",
                
                # --- PIPELINE HYPERPARAMETERS ---
                filter_threshold=-1.0,  # -1.0 generates the master JSON for simulation
                filter_sample_size=5,
                cluster_size_limit=-1,
                block_size=15,
                min_profile_size=0,
                cluster_ratio=divisor, 
                verbose=False 
            )

            # Dynamically name the output file
            suffix = "_projection" if force_proj else ""
            output_filename = f"src/evaluation/sim_logs/imdb_master_log_ratio_{ratio}{suffix}.json"

            # Export simulation data
            export_simulation_data(
                result=result, 
                table_a=df_a, 
                table_b=df_b, 
                filepath=output_filename,
                num_samples=3
            )

if __name__ == "__main__":
    evaluate_imdb_pairwise()