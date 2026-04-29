import pandas as pd
import os

# Define the paths to your aggregated result CSVs
datasets = {
    'Emails': 'src/results/emails_projection_aggregated_results.csv',
    'StackOverflow': 'src/results/stackoverflow_no_desc_projection_aggregated_results.csv',
    'IMDB': 'src/results/imdb_projection_aggregated_results.csv'
}

def analyze_optimal_tradeoffs(dataset_name, filepath, max_f1_loss=2.5):
    """
    Finds the absolute max F1, then searches for cheaper configurations 
    that stay within `max_f1_loss` points of the maximum.
    """
    df = pd.read_csv(filepath)
    
    # 1. Find the Absolute Maximum F1
    max_f1 = df['F1 (%)'].max()
    
    # If multiple configs achieve the exact max F1, take the cheapest one
    max_f1_row = df[df['F1 (%)'] == max_f1].sort_values('Total Tokens').iloc[0]
    
    print(f"[{dataset_name}]")
    print(f"  Absolute Max F1 : {max_f1:.2f}% (Ratio: {max_f1_row['Ratio']}, Thresh: {max_f1_row['Threshold']})")
    print(f"  Max F1 Cost     : {max_f1_row['Total Tokens']:,} tokens")
    
    # 2. Find Candidates on the Pareto Frontier
    # (Cheaper than max_f1_row, and within acceptable F1 loss)
    candidates = df[
        (df['F1 (%)'] >= (max_f1 - max_f1_loss)) & 
        (df['Total Tokens'] < max_f1_row['Total Tokens'])
    ]
    
    if candidates.empty:
        print(f"  Optimal Choice  : Stick with Absolute Max. No cheaper alternatives within {max_f1_loss} F1 points.\\n")
        return

    # Sort candidates by cost (ascending), then F1 (descending)
    candidates = candidates.sort_values(['Total Tokens', 'F1 (%)'], ascending=[True, False])
    
    # Filter strictly for pareto efficiency (must yield higher F1 to justify higher cost)
    best_f1_so_far = 0
    print(f"  Optimal Alternatives (Max {max_f1_loss} pt drop):")
    
    for _, row in candidates.iterrows():
        if row['F1 (%)'] > best_f1_so_far:
            best_f1_so_far = row['F1 (%)']
            
            savings_pct = ((max_f1_row['Total Tokens'] - row['Total Tokens']) / max_f1_row['Total Tokens']) * 100
            f1_drop = max_f1 - row['F1 (%)']
            
            print(f"    -> {row['F1 (%)']:.2f}% F1 (-{f1_drop:.2f} pts) | Ratio: {row['Ratio']} | Thresh: {row['Threshold']}")
            print(f"       Cost: {row['Total Tokens']:,} tokens ({savings_pct:.1f}% savings)")
            
    print() # newline spacing

if __name__ == "__main__":
    print("=== Optimal Hyperparameter Analysis ===\\n")
    
    for name, filepath in datasets.items():
        if os.path.exists(filepath):
            # You can adjust max_f1_loss to be more or less aggressive 
            # with how much accuracy you are willing to trade for tokens.
            analyze_optimal_tradeoffs(name, filepath, max_f1_loss=2.5)
        else:
            print(f"Skipping {name}: Could not find {filepath}\\n")