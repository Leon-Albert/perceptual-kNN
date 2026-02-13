import pandas as pd
import networkx as nx
import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_knn_graphml(input_path, output_path, k=5, large_val=1e10):

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if 'row_id' in df.columns:
        node_ids = df['row_id'].values
        features = df.drop(columns=['row_id']).values
    else:
        node_ids = df.index.values
        features = df.values

    print("Handling NaNs and Infs...")
    features = np.nan_to_num(
        features, 
        nan=0.0, 
        posinf=large_val, 
        neginf=-large_val
    )

    print(f"Computing KNN (k={k}) on {len(features)} samples...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric='minkowski', p=2, n_jobs=-1)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    G = nx.Graph()
    G.add_nodes_from(node_ids)

    print("Building edges...")
    for i in range(len(node_ids)):
        source = node_ids[i]
        for j in range(1, k + 1):
            target = node_ids[indices[i, j]]
            dist = float(distances[i, j])
            
            G.add_edge(source, target, distance=dist)

    print(f"Saving graph to {output_path}...")
    nx.write_graphml(G, output_path)
    print("Done.")

if __name__ == "__main__":
    DATA_DIR = os.path.join(sys.path[0], "data", "precompute_S")
    INPUT_FILE = os.path.join(DATA_DIR, "S_dataset.parquet")
    OUTPUT_FILE = os.path.join(DATA_DIR, "knn_graph.graphml")
    
    generate_knn_graphml(INPUT_FILE, OUTPUT_FILE, k=35, large_val=1e10)