import torch
import pickle
from torch_geometric.utils import k_hop_subgraph
import json

def load_data(file_path):
    return torch.load(file_path)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def clean_text(text, dataset):
    if dataset in ['yelp', 'google']:
        return text.split('"summarization": "')[1].rsplit('"', 1)[0].strip()
    elif dataset == 'amazon':
        json_data = json.loads(text.replace('\n', ''))
        return json_data['summarization'].rsplit('"', 1)[0].strip()

def extract_k_hop_subgraph(pyg_data, user_node, item_node, k):
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes

    user_nodes, user_edge_index, _, _ = k_hop_subgraph(
        node_idx=user_node, 
        num_hops=k, 
        edge_index=edge_index, 
        relabel_nodes=False,
        num_nodes=num_nodes
    )
    # print(f"Using user node {user_node} to retrieve item nodes: {user_nodes}")

    item_nodes, item_edge_index, _, _ = k_hop_subgraph(
        node_idx=item_node, 
        num_hops=k, 
        edge_index=edge_index, 
        relabel_nodes=False,
        num_nodes=num_nodes
    )
    # print(f"Using item node {item_node} to retrieve user nodes: {item_nodes}")

    combined_nodes = torch.unique(torch.cat([user_nodes, item_nodes]))
    combined_edge_index = torch.cat([user_edge_index, item_edge_index], dim=1)
    
    combined_edge_index = torch.unique(combined_edge_index, dim=1)
    
    return combined_nodes, combined_edge_index