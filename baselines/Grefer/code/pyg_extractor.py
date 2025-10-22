import torch
import argparse
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch_geometric.data import Data
from dataset import KHopSubgraphDataset
from converter import TextModel
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_k_hop_subgraph, load_data, load_pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to PyG format')
    parser.add_argument('--dataset', type=str, choices=['yelp', 'google', 'amazon'], required=True,
                        help='Dataset to convert (yelp, google, or amazon)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True,
                        help='Data split to convert (trn, val, or tst)')
    parser.add_argument('--k', type=int, default=2, help='Hop count')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for spectral clustering')
    return parser.parse_args()

def count_users_items(subgraph_nodes, pyg_data):
    user_mask = torch.isin(subgraph_nodes, torch.tensor(list(pyg_data.user_id_to_node.values())))
    user_count = torch.sum(user_mask).item()
    item_count = len(subgraph_nodes) - user_count
    return user_count, item_count

def spectral_clustering(data, n_clusters):
    """
    Perform spectral clustering based on node features of a graph.

    Parameters:
    - data (torch_geometric.data.Data): Graph data object containing node features `x`.
    - n_clusters (int): Number of clusters to form.

    Returns:
    - numpy.ndarray: Array of cluster labels for each node.
    """
    x = data.x.cpu().numpy()

    # Calculate the Euclidean distance matrix for features
    distances = pdist(x, metric='euclidean')
    # Convert distances to a similarity matrix using a Gaussian kernel
    similarity_matrix = np.exp(-squareform(distances)**2 / (2. * np.std(distances)**2))

    # Spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(similarity_matrix)

    return labels

def process_data(pyg_data, pkl_data, k, n_clusters):
    results = []
    sentence_model = TextModel(encoder='SentenceBert')
    sentence_model = sentence_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Create a progress bar
    progress_bar = tqdm(pkl_data.iterrows(), total=len(pkl_data), desc="Processing data")

    for _, row in progress_bar:
        user = row['uid']
        item = row['iid']
        
        if user in pyg_data.user_id_to_node and item in pyg_data.item_id_to_node:
            user_node = pyg_data.user_id_to_node[user]
            item_node = pyg_data.item_id_to_node[item]
            
            subgraph_nodes, subgraph_edge_index = extract_k_hop_subgraph(pyg_data, user_node, item_node, k)
            user_count, item_count = count_users_items(subgraph_nodes, pyg_data)
            
            # Perform community detection
            subgraph_data = Data(x=pyg_data.x[subgraph_nodes], edge_index=subgraph_edge_index)
            community_labels = spectral_clustering(subgraph_data, n_clusters)
            
            # Calculate semantic similarity
            with torch.no_grad():
                explanation_embedding = sentence_model(row['explanation']).cpu()
            cluster_embeddings = [pyg_data.x[subgraph_nodes[community_labels == i]].mean(dim=0).cpu().numpy() for i in range(n_clusters)]
            similarities = cosine_similarity(explanation_embedding.reshape(1, -1), cluster_embeddings)[0]
            most_similar_cluster = np.argmax(similarities)
            
            results.append({
                'nodes': subgraph_nodes,
                'edge_index': subgraph_edge_index,
                'user': user,
                'item': item,
                'user_count': user_count,
                'item_count': item_count,
                'community_labels': community_labels,
                'most_similar_cluster': most_similar_cluster
            })
            
            # Update progress bar description
            progress_bar.set_postfix(user=user, item=item, users=user_count, items=item_count)
    
    return results

def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} {args.split} data...")
    
    pyg_data = load_data(f'data/{args.dataset}/data_{args.split}.pt')
    pkl_data = load_pickle(f'data/{args.dataset}/{args.split}.pkl')
    
    results = process_data(pyg_data, pkl_data, args.k, args.n_clusters)
    
    dataset = KHopSubgraphDataset(results, pyg_data)
    
    save_path = f'data/{args.dataset}/khop_subgraphs_{args.split}.pt'
    torch.save(dataset, save_path)
    
    print(f"Processed {len(results)} subgraphs for {args.split} data")
    print(f"Dataset saved to {save_path}")

    # Print sample results
    sample_subgraph = dataset[0]
    print(f"Sample subgraph:")
    print(f"  User: {sample_subgraph.user}, Item: {sample_subgraph.item}")
    print(f"  Number of nodes: {sample_subgraph.num_nodes}")
    print(f"  Number of edges: {sample_subgraph.num_edges}")
    print(f"  Community labels: {sample_subgraph.community_labels}")

if __name__ == "__main__":
    main()