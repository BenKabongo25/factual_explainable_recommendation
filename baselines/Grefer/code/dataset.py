from torch_geometric.data import Data, Dataset
import torch

class KHopSubgraphDataset(Dataset):
    def __init__(self, subgraphs, original_data):
        super().__init__()
        self.subgraphs = subgraphs
        self.original_data = original_data

    def len(self):
        return len(self.subgraphs)

    def get(self, idx):
        subgraph = self.subgraphs[idx]
        nodes, edge_index = subgraph['nodes'], subgraph['edge_index']
        
        # Create a new Data object for the subgraph
        data = Data(
            x=self.original_data.x[nodes],
            edge_index=edge_index,
            nodes=nodes,
            user=subgraph['user'],
            item=subgraph['item'],
            community_labels=subgraph['community_labels']
        )
        return data

class DataMapper:
    def __init__(self, pt_file_path):
        self.original_data = torch.load(pt_file_path)
        self.num_users = len(self.original_data.user_id_to_node)

    def get_user_raw_text(self, dgl_user_id):
        return self.original_data.raw_texts[dgl_user_id]

    def get_item_title(self, dgl_item_id):
        return self.original_data.item_titles[dgl_item_id]
    
    def get_item_raw_text(self, dgl_item_id):
        return self.original_data.raw_texts[dgl_item_id + self.num_users]