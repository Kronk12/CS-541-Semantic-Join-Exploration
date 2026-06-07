import pandas as pd
import os

# Define the paths to your aggregated result CSVs
LOG_DIR = 'src/evaluation/logs'

datasets = [
    ('Emails', 'emails_aggregated_results.csv', 'emails_projection_aggregated_results.csv'),
    ('StackOverflow', 'stackoverflow_no_desc_aggregated_results.csv', 'stackoverflow_no_desc_projection_aggregated_results.csv'),
    ('IMDB', 'imdb_aggregated_results.csv', 'imdb_projection_aggregated_results.csv')
]

def analyze_optimal_tradeoffs(dataset_name, std_file, proj_file, max_f1_loss=2.5):
    """
    Finds the absolute max F1 across both standard and projection states, 
    then searches for cheaper configurations that stay within `max_f1_loss`.
    """
    dfs = []
    if os.path.exists(os.path.join(LOG_DIR, std_file)):
        dfs.append(pd.read_csv(os.path.join(LOG_DIR, std_file)))
    if os.path.exists(os.path.join(LOG_DIR, proj_file)):
        dfs.append(pd.read_csv(os.path.join(LOG_DIR, proj_file)))
        
    if not dfs:
        print(f"Skipping {dataset_name}: No result files found.\n")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Find the Absolute Maximum F1
    max_f1 = df['F1 (%)'].max()
    
    # If multiple configs achieve the exact max F1, take the cheapest one
    max_f1_row = df[df['F1 (%)'] == max_f1].sort_values('Total Tokens').iloc[0]
    
    print(f"[{dataset_name}]")
    print(f"  Absolute Max F1 : {max_f1:.2f}% (Subgroup: {max_f1_row['Subgroup']}, Ratio: {max_f1_row['Ratio']}, Thresh: {max_f1_row['Threshold']})")
    print(f"  Max F1 Cost     : {max_f1_row['Total Tokens']:,} tokens")
    
    # 2. Find Candidates on the Pareto Frontier
    candidates = df[
        (df['F1 (%)'] >= (max_f1 - max_f1_loss)) & 
        (df['Total Tokens'] < max_f1_row['Total Tokens'])
    ]
    
    if candidates.empty:
        print(f"  Optimal Choice  : Stick with Absolute Max. No cheaper alternatives within {max_f1_loss} F1 points.\n")
        return

    # Sort candidates by cost (ascending), then F1 (descending)
    candidates = candidates.sort_values(['Total Tokens', 'F1 (%)'], ascending=[True, False])
    
    # Filter strictly for pareto efficiency
    best_f1_so_far = 0
    print(f"  Optimal Alternatives (Max {max_f1_loss} pt drop):")
    
    for _, row in candidates.iterrows():
        if row['F1 (%)'] > best_f1_so_far:
            best_f1_so_far = row['F1 (%)']
            
            savings_pct = ((max_f1_row['Total Tokens'] - row['Total Tokens']) / max_f1_row['Total Tokens']) * 100
            f1_drop = max_f1 - row['F1 (%)']
            
            print(f"    -> {row['F1 (%)']:.2f}% F1 (-{f1_drop:.2f} pts) | {row['Subgroup']} | Ratio: {row['Ratio']} | Thresh: {row['Threshold']}")
            print(f"       Cost: {row['Total Tokens']:,} tokens ({savings_pct:.1f}% savings)")
            
    print()

if __name__ == "__main__":
    print("=== Optimal Hyperparameter Analysis ===\n")
    
    for name, std, proj in datasets:
        analyze_optimal_tradeoffs(name, std, proj, max_f1_loss=2.5)