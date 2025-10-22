import os
import dgl
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from utils import load_pickle, load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to DGL format')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to convert')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Directory where the dataset is stored')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the converted data')
    parser.add_argument('--split', type=str, choices=['train', 'eval', 'test'], required=True,
                        help='Data split to convert (train, eval, or test)')
    return parser.parse_args()

def create_dgl_graph(pyg_data, df_data):
    num_users = len(pyg_data.user_id_to_node)
    num_items = len(pyg_data.item_id_to_node)
    
    g = dgl.heterograph({
        ('user', 'have_bought', 'item'): ([], []),
        ('item', 'bought_by', 'user'): ([], []),
        ('user', 'likes', 'item'): ([], [])
    }, num_nodes_dict={'user': num_users, 'item': num_items})

    print(f"Number of rows in df_data: {len(df_data)}")

    # Create likes_edges set
    likes_edges = set()
    for _, row in tqdm(df_data.iterrows(), desc="Processing df_data", total=len(df_data)):
        if row['user_id'] in pyg_data.user_id_to_node and row['item_id'] in pyg_data.item_id_to_node:
            user_idx = pyg_data.user_id_to_node[row['user_id']]
            item_idx = pyg_data.item_id_to_node[row['item_id']] - num_users
            likes_edges.add((user_idx, item_idx))
        else:
            print(row['user_id'], row['item_id'])

    print(f"Number of likes edges: {len(likes_edges)}")

    buys_edges = set()
    
    # Process all edges from pyg_data
    for src, dst in tqdm(pyg_data.edge_index.t().tolist(), desc="Processing edges"):
        if src < num_users:  # src is a user node
            user_idx = src
            item_idx = dst - num_users
            buys_edges.add((user_idx, item_idx))

    # Remove likes edges from buys edges
    buys_edges -= likes_edges

    # Create final edge lists
    buys_src, buys_dst = zip(*buys_edges) if buys_edges else ([], [])
    likes_src, likes_dst = zip(*likes_edges) if likes_edges else ([], [])
    bought_by_src, bought_by_dst = buys_dst, buys_src

    # Add edges to the graph
    g.add_edges(buys_src, buys_dst, etype='buys')
    g.add_edges(bought_by_src, bought_by_dst, etype='bought_by')
    g.add_edges(likes_src, likes_dst, etype='likes')

    print(f"Processed likes edges: {len(likes_src)}")
    print(f"Processed buys edges: {len(buys_src)}")
    print(f"Processed bought_by edges: {len(bought_by_src)}")

    # Add node features
    g.nodes['user'].data['feat'] = pyg_data.x[:num_users]
    g.nodes['item'].data['feat'] = pyg_data.x[num_users:]

    return g

def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} {args.split} data...")
    
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
    df = pd.read_csv(df_data_path)
    df['user_id'] = df['user_id'].map(inv_user_map)
    df['item_id'] = df['item_id'].map(inv_item_map)
    print(df.head(5).to_string())
    
    dgl_graph = create_dgl_graph(pyg_data, df)
    
    # Save only the DGL graph
    dgl.save_graphs(os.path.join(args.output_dir, f'{args.split}_graph.bin'), [dgl_graph])
    #dgl.save_graphs(f'./data/{args.dataset}/{args.split}_graph.bin', [dgl_graph])

    # also save a copy to pagelink
    #dgl.save_graphs(f'./PaGE-Link/datasets/{args.dataset}_{args.split}.bin', [dgl_graph])
    
    print(f"Processed graph for {args.split} data")

    print(f"Graph statistics:")
    print(f"  Number of users: {dgl_graph.num_nodes('user')}")
    print(f"  Number of items: {dgl_graph.num_nodes('item')}")
    print(f"  Number of edges:")
    for etype in dgl_graph.etypes:
        print(f"    {etype}: {dgl_graph.num_edges(etype=etype)}")
    print(f"  Node feature dimensions:")
    print(f"    User: {dgl_graph.nodes['user'].data['feat'].shape[1]}")
    print(f"    Item: {dgl_graph.nodes['item'].data['feat'].shape[1]}")

if __name__ == "__main__":
    main()

# python dgl_extractor.py --split trn --dataset yelp