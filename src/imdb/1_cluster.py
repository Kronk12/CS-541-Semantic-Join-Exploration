import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
import time

def cluster_tables(num_clusters=2):
    print(f"--- Starting Clustering Pipeline (k={num_clusters}) ---")
    total_start_time = time.time()
    
    # Load your tables
    table_a = pd.read_csv('data/table_a.csv')
    table_b = pd.read_csv('data/table_b.csv')

    # Initialize the embedding model
    print("Initializing embedding model...")
    model = SentenceTransformer('distilbert-base-uncased-finetuned-sst-2-english')

    # Generate embeddings for both tables
    print("Generating embeddings for Table A and Table B...")
    embed_start_time = time.time()
    embeddings_a = model.encode(table_a['review'].str[:512].tolist())
    embeddings_b = model.encode(table_b['review'].str[:512].tolist())
    embed_time = time.time() - embed_start_time
    print(f"  -> Embeddings generated in {embed_time:.2f} seconds.")

    # Perform K-means clustering 
    print("Running K-Means algorithm...")
    kmeans_start_time = time.time()
    kmeans_a = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    table_a['cluster'] = kmeans_a.fit_predict(embeddings_a)

    kmeans_b = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    table_b['cluster'] = kmeans_b.fit_predict(embeddings_b)
    kmeans_time = time.time() - kmeans_start_time
    print(f"  -> Clustering finished in {kmeans_time:.2f} seconds.")

    # Save the clustered results
    table_a.to_csv('data/table_a_clustered_distilbert.csv', index=False)
    table_b.to_csv('data/table_b_clustered_distilbert.csv', index=False)

    total_time = time.time() - total_start_time

    print(f"\n--- Clustering Complete! ---")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Table A clusters: {table_a['cluster'].value_counts().to_dict()}")
    print(f"Table B clusters: {table_b['cluster'].value_counts().to_dict()}")
    
    # Preview a cluster
    print("\nPreviewing Table A, Cluster 0 (First 2 reviews):")
    print(table_a[table_a['cluster'] == 0]['review'].str[:100].tolist()[:2])

if __name__ == "__main__":
    cluster_tables(num_clusters=5)