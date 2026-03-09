import pandas as pd
import os

def evaluate_cluster_purity(file_path, table_name):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    if 'cluster' not in df.columns or 'sentiment' not in df.columns:
        print(f"Error: {file_path} must contain 'cluster' and 'sentiment' columns.")
        return

    print(f"\n=== Evaluating {table_name} ===")
    
    # Group by cluster and sentiment to get the raw counts
    distribution = df.groupby(['cluster', 'sentiment']).size().unstack(fill_value=0)
    
    # Get the dynamic list of sentiment labels (e.g., ['negative', 'positive'])
    sentiments = distribution.columns.tolist()
    
    # Print dynamic header
    header_counts = " | ".join([f"{str(s).title():<10}" for s in sentiments])
    print(f"{'Cluster':<8} | {header_counts} | {'Purity':<8} | {'Dominant':<10}")
    print("-" * (40 + 13 * len(sentiments)))

    total_rows = len(df)
    weighted_purity_sum = 0

    # Iterate through each cluster to calculate metrics
    for cluster_id, row in distribution.iterrows():
        total_in_cluster = row.sum()
        dominant_sentiment = row.idxmax()
        dominant_count = row.max()
        
        # Purity is the percentage of the most frequent label in the cluster
        purity = (dominant_count / total_in_cluster) * 100 if total_in_cluster > 0 else 0
        weighted_purity_sum += dominant_count
        
        # Format the row dynamically
        counts_str = " | ".join([f"{row[s]:<10}" for s in sentiments])
        print(f"{cluster_id:<8} | {counts_str} | {purity:>7.2f}% | {str(dominant_sentiment).title():<10}")

    # Calculate overall purity across the whole table
    overall_purity = (weighted_purity_sum / total_rows) * 100 if total_rows > 0 else 0
    print("-" * (40 + 13 * len(sentiments)))
    print(f"Overall Clustering Purity for {table_name}: {overall_purity:.2f}%\n")

def main():
    evaluate_cluster_purity('data/table_a_clustered_distilbert.csv', 'Table A')
    evaluate_cluster_purity('data/table_b_clustered_distilbert.csv', 'Table B')

if __name__ == "__main__":
    main()