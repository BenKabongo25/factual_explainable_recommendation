import torch
import os
import pandas as pd
import json
from torch_geometric.data import Data
import argparse
import os
import tqdm
import pickle
from model import TextModel
from utils import clean_text

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to PyG format')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to convert')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Directory where the dataset is stored')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the converted data')
    parser.add_argument('--profiles_dir', type=str, default='data',
                        help='Directory where user and item profiles are stored')
    parser.add_argument('--split', type=str, choices=['train', 'eval', 'test'], required=True,
                        help='Data split to convert (train, eval, or test)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(args.dataset_dir, f'{args.split}_data.csv'))

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

    for split_df in split_dfs:
        split_df['user_id'] = split_df['user_id'].map(inv_user_map)
        split_df['item_id'] = split_df['item_id'].map(inv_item_map)
    df['user_id'] = df['user_id'].map(inv_user_map)
    df['item_id'] = df['item_id'].map(inv_item_map)
    print(df.head(5).to_string())

    user_id_to_node = {id: i for i, id in enumerate(user_map)}
    item_id_to_node = {id: i + len(user_id_to_node) for i, id in enumerate(item_map)}

    # Load user profiles
    user_profiles = pd.read_csv(os.path.join(args.profiles_dir, 'user_summaries.csv'))
    user_profiles = user_profiles[['user_id', 'summary_text']]
    user_profiles['user_id'] = user_profiles['user_id'].map(inv_user_map)
    print(user_profiles.head(5).to_string())
    user_profiles = user_profiles.set_index('user_id').to_dict()['summary_text']
    for uid in user_map.keys():
        if (uid not in user_profiles or 
            not isinstance(user_profiles[uid], str) or 
            not user_profiles[uid].strip()):
            user_profiles[uid] = ""

        if uid < 10:
            print(f"User ID: {uid}, Summary: {user_profiles[uid]}")

    # Load item profiles
    item_profiles = pd.read_csv(os.path.join(args.profiles_dir, 'item_summaries.csv'))
    item_profiles = item_profiles[['item_id', 'title', 'description', 'summary_text']]
    item_profiles['item_id'] = item_profiles['item_id'].map(inv_item_map)
    print(item_profiles.head(5).to_string())

    item_titles = item_profiles.set_index('item_id').to_dict()['title']
    item_descriptions = item_profiles.set_index('item_id').to_dict()['description']
    item_profiles = item_profiles.set_index('item_id').to_dict()['summary_text']

    for iid in item_map.keys():
        if (iid not in item_profiles or 
            not isinstance(item_profiles[iid], str) or 
            not item_profiles[iid].strip()):
            description = item_descriptions.get(iid, "")
            if isinstance(description, str) and description.strip():
                item_profiles[iid] = description
            else:
                title = item_titles.get(iid, "")
                if isinstance(title, str) and title.strip():
                    item_profiles[iid] = title
                else:
                    item_profiles[iid] = ""
        
        if iid < 10:
            print(f"Item ID: {iid}, Title: {item_titles.get(iid, '')}, Description: {item_descriptions.get(iid, '')}, Summary: {item_profiles[iid]}")
    
    # Create ordered lists of texts based on user_id_to_node and item_id_to_node
    ordered_user_texts = [user_profiles[uid] for uid in user_id_to_node.keys()]
    ordered_item_texts = [item_profiles[iid] for iid in item_id_to_node.keys()]

    items_with_title = sum(1 for title in item_titles.values() if isinstance(title, str) and title.strip())
    items_without_title = len(item_titles) - items_with_title

    print(f"Items with title: {items_with_title}")
    print(f"Items without title: {items_without_title}")

    # Create bidirectional edge_index
    user_nodes = [user_id_to_node[u] for u in df['user_id']]
    item_nodes = [item_id_to_node[i] for i in df['item_id']]
    edge_index = torch.tensor([
        user_nodes + item_nodes, 
        item_nodes + user_nodes
    ], dtype=torch.long)

    # item_combined_texts = [f"Title: {title}\nSummary: {text}" for title, text in zip(item_titles, item_texts)]
    
    all_texts = ordered_user_texts + ordered_item_texts

    text_model = TextModel(args.text_encoder)

    text_model = text_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    text_features = []

    batch_size = 128
    for i in tqdm.tqdm(range(0, len(all_texts), batch_size), desc="Processing texts"):
        batch = all_texts[i:i+batch_size]
        with torch.no_grad():
            batch_features = text_model(batch).cpu()
        text_features.append(batch_features)

    text_features = torch.cat(text_features, dim=0)

    data = Data(
        edge_index=edge_index,
        num_nodes=len(user_id_to_node) + len(item_id_to_node),
        raw_texts=all_texts,
        x=text_features,
        user_id_to_node=user_id_to_node,
        item_id_to_node=item_id_to_node,
        item_titles=item_titles
    )

    save_path = os.path.join(args.output_dir, f'data_{args.split}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)

    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()
