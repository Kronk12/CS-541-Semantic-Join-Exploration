import pandas as pd
import json
import os

def evaluate_filtering_step():
    print("--- Evaluating Cluster Filtering Step ---")
    
    # 1. Load the Data
    if not os.path.exists('data/table_a_clustered_distilbert.csv') or not os.path.exists('data/table_b_clustered_distilbert.csv'):
        print("Error: Missing clustered tables.")
        return
        
    table_a = pd.read_csv('data/table_a_clustered_distilbert.csv')
    table_b = pd.read_csv('data/table_b_clustered_distilbert.csv')
    
    with open('data/valid_cluster_pairs_distilbert.json', 'r') as f:
        valid_pairs_list = json.load(f)
        # Convert to a set of tuples for fast lookup
        valid_pairs = set(tuple(pair) for pair in valid_pairs_list)

    # 2. Track Search Space and Matches
    total_comparisons_before = len(table_a) * len(table_b)
    total_comparisons_after = 0
    
    total_ground_truth_matches = 0
    matches_kept = 0
    matches_dropped = 0

    # 3. Iterate through all cluster combinations
    clusters_a = table_a['cluster'].unique()
    clusters_b = table_b['cluster'].unique()

    for ca in clusters_a:
        rows_a = table_a[table_a['cluster'] == ca]
        for cb in clusters_b:
            rows_b = table_b[table_b['cluster'] == cb]
            
            # Calculate total cross-product pairs for this cluster combination
            pair_count = len(rows_a) * len(rows_b)
            
            # Find actual ground truth matches in this cluster combination
            actual_matches_in_cluster_pair = 0
            for _, ra in rows_a.iterrows():
                for _, rb in rows_b.iterrows():
                    if ra['sentiment'] == rb['sentiment']:
                        actual_matches_in_cluster_pair += 1
                        total_ground_truth_matches += 1

            # Check if this cluster pair survived the filter
            if (ca, cb) in valid_pairs:
                total_comparisons_after += pair_count
                matches_kept += actual_matches_in_cluster_pair
            else:
                matches_dropped += actual_matches_in_cluster_pair

    # 4. Calculate Final Metrics
    search_space_reduction = (1 - (total_comparisons_after / total_comparisons_before)) * 100
    filter_recall = (matches_kept / total_ground_truth_matches) * 100 if total_ground_truth_matches > 0 else 0

    # 5. Output the Report
    print(f"\n--- Search Space Reduction ---")
    print(f"Original Comparisons (Naive) : {total_comparisons_before}")
    print(f"Comparisons Remaining (Block): {total_comparisons_after}")
    print(f"Workload Reduction           : {search_space_reduction:.2f}%\n")

    print(f"--- Recall at Filtering Stage ---")
    print(f"Total Actual Matches         : {total_ground_truth_matches}")
    print(f"Matches Kept in Valid Pairs  : {matches_kept}")
    print(f"Matches Accidentally Dropped : {matches_dropped}")
    print(f"Maximum Possible Final Recall: {filter_recall:.2f}%")

if __name__ == "__main__":
    evaluate_filtering_step()