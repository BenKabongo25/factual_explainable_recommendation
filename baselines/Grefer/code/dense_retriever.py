import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import extract_k_hop_subgraph, load_data, load_pickle
from tqdm import tqdm
from dataset import DataMapper
import json
import os
import pandas as pd
import torch

class DenseRetriever(nn.Module):
    def __init__(self, pruning_score):
        super(DenseRetriever, self).__init__()
        self.pruning_score = pruning_score

    def forward(self, x):
        return x

    def cosine_similarity(self, x, y):
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

    def retrieve_topk(self, pyg_data, pd_data, data_mapper, topk):
        self.eval()

        # count the avg users and items retrieved
        avg_users_retrieved = 0
        avg_items_retrieved = 0

        retrieval_results = {}

        progress_bar = tqdm(pd_data.iterrows(), total=len(pd_data), desc="Dense Retrieval")

        for _, row in progress_bar:
            user_idx = pyg_data.user_id_to_node.get(row['user_id'])
            item_idx = pyg_data.item_id_to_node.get(row['item_id'])
            
            if user_idx is None or item_idx is None:
                continue

            _, subgraph_edge_index = extract_k_hop_subgraph(pyg_data, user_idx, item_idx, 2)
            
            # Calculate cosine similarity for users who bought the current item
            user_feat = pyg_data.x[user_idx]
            num_users = len(pyg_data.user_id_to_node)
            subgraph_edge_index = subgraph_edge_index[:, (subgraph_edge_index[0] < num_users) | (subgraph_edge_index[1] >= num_users)]
            users_who_bought_item = set(subgraph_edge_index[0][subgraph_edge_index[1] == item_idx].tolist())
            users_who_bought_item.discard(user_idx)
            user_similarities = [(other_user, self.cosine_similarity(user_feat, pyg_data.x[other_user]).item()) for other_user in users_who_bought_item]
            topk_users = sorted(user_similarities, key=lambda x: x[1], reverse=True)[:topk]

            # Calculate cosine similarity for items bought by the current user
            item_feat = pyg_data.x[item_idx]
            items_bought_by_user = set(subgraph_edge_index[1][subgraph_edge_index[0] == user_idx].tolist())
            items_bought_by_user.discard(item_idx)
            item_similarities = [(other_item, self.cosine_similarity(item_feat, pyg_data.x[other_item]).item()) for other_item in items_bought_by_user]
            topk_items = sorted(item_similarities, key=lambda x: x[1], reverse=True)[:topk]

            # pruning based on the similarity score
            topk_users = [user for user, sim in topk_users if sim >= self.pruning_score]
            topk_items = [item for item, sim in topk_items if sim >= self.pruning_score]
        
            avg_users_retrieved += len(topk_users)
            avg_items_retrieved += len(topk_items)

            # Store retrieval results
            retrieval_results[f"{user_idx}-{item_idx}"] = {
                "topk_user_ids": topk_users,
                "topk_item_ids": topk_items,
                "topk_user_raw_texts": [data_mapper.get_user_raw_text(user) for user in topk_users],
                "topk_item_raw_texts": [data_mapper.get_item_raw_text(item - data_mapper.num_users) for item in topk_items],
                "topk_user_similarities": [sim for sim in topk_users],
                "topk_item_similarities": [sim for sim in topk_items]
            }

        print(f"Average users retrieved: {avg_users_retrieved / len(pd_data)}")
        print(f"Average items retrieved: {avg_items_retrieved / len(pd_data)}")    

        return retrieval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve top-k similar users and items')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Directory where the dataset is stored')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the converted data')
    parser.add_argument('--split', type=str, choices=['train', 'eval', 'test'], required=True,
                        help='Data split to use (train, eval, or test)')
    parser.add_argument('--topk', type=int, default=5, help='Number of top-k similar nodes to retrieve')
    parser.add_argument('--pruning_score', type=float, default=0, help='Pruning score for similarity')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} {args.split} data...")
    
    #pyg_data = load_data(f'data/{args.dataset}/data_{args.split}.pt')  # Load PyG data
    #pd_data = load_pickle(f'data/{args.dataset}/{args.split}.pkl')

    pyg_data_path = os.path.join(args.output_dir, f"data_{args.split}.pt")
    pyg_data = torch.load(pyg_data_path)

    splits = ['train', 'eval', 'test']
    split_dfs = []
    for split in splits:
        split_path = os.path.join(args.dataset_dir, f'{split}_data.csv')
        split_dfs.append(pd.read_csv(split_path))
    data_df = pd.concat(split_dfs, ignore_index=True)
    all_users = data_df['user_id'].unique().tolist()
    all_items = data_df['item_id'].unique().tolist()

    user_map = {uid: u for uid, u in enumerate(all_users)}
    item_map = {iid: i for iid, i in enumerate(all_items)}

    inv_user_map = {u: uid for uid, u in user_map.items()}
    inv_item_map = {i: iid for iid, i in item_map.items()}

    df_data_path = os.path.join(args.dataset_dir, f"{args.split}_data.csv")
    pd_data = pd.read_csv(df_data_path)
    pd_data['user_id'] = pd_data['user_id'].map(inv_user_map)
    pd_data['item_id'] = pd_data['item_id'].map(inv_item_map)
    print(pd_data.head(5).to_string())

    data_mapper = DataMapper(os.path.join(args.output_dir, f"data_{args.split}.pt"))
    
    retriever = DenseRetriever(args.pruning_score)
    retrieval_results = retriever.retrieve_topk(pyg_data, pd_data, data_mapper, args.topk)

    # Save retrieval results
    with open(os.path.join(args.output_dir, f'dense_retrieval_results_{args.split}.json'), 'w') as f:
        json.dump(retrieval_results, f)

    print(f"Retrieval results saved to {os.path.join(args.output_dir, f'dense_retrieval_results_{args.split}.json')}")

if __name__ == "__main__":
    main()

# python Retriever/dense_retriever.py --dataset yelp --split trn --topk 5