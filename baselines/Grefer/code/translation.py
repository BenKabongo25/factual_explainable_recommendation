import argparse
from dataset import DataMapper
import json
import pickle
import os
from utils import load_pickle, load_data
from model import TextModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Flatten retrieval results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., yelp)')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Directory where the dataset is stored')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the converted data')
    parser.add_argument('--profiles_dir', type=str, default='data',
                        help='Directory where user and item profiles are stored')
    parser.add_argument('--split', type=str, required=True, help='Data split to use (e.g., train)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    parser.add_argument('--k', type=int, default=2, help='Number of paths, users, and items to select for each user-item pair')
    parser.add_argument('--saved_model_name', type=str, default='')
    return parser.parse_args()

def flatten_pagelink_retrieval_results(pagelink_retrieval_results, mapper, k):
    flattened_results = {}
    for key, value in pagelink_retrieval_results.items():
        top_paths = value[:k]
        path_prompts = []
        
        prompt_template = "For the given user-item pair, here are several related paths connecting users and items through their interactions:"
        path_prompts = []
        for idx, path in enumerate(top_paths, 1):
            path_prompt = []
            for i, edge in enumerate(path):
                if edge[0][0] == 'user':
                    user_text = mapper.get_user_raw_text(edge[1])
                    path_prompt.append(f"User (Profile: {user_text})")
                elif edge[0][0] == 'item':
                    item_text = mapper.get_item_raw_text(edge[1])
                    path_prompt.append(f"Item (Profile: {item_text})")
                
                if i < len(path) - 1:
                    next_edge = path[i + 1]
                    if edge[0][0] == 'user' and next_edge[0][0] == 'item':
                        path_prompt.append("buys")
                    elif edge[0][0] == 'item' and next_edge[0][0] == 'user':
                        path_prompt.append("bought by")

            if path and path[-1][0][-1] == 'item':
                last_item_id = path[-1][2]
                last_item_text = mapper.get_item_raw_text(last_item_id)
                path_prompt.append("buys")
                path_prompt.append(f"Item (Profile: {last_item_text})")
            
                path_text = " -> ".join(path_prompt)
                path_prompts.append(f"{idx}. {path_text}")
            
        user_id = key[0][1]
        item_id = key[1][1]
        flattened_results[(user_id, item_id)] = prompt_template + " " + " ".join(path_prompts)
    
    return flattened_results

def flatten_dense_retrieval_results(dense_retrieval_results, mapper, k):
    flattened_results = {}
    for key, value in dense_retrieval_results.items():
        user_item_tuple = tuple(map(int, key.split('-')))
        user_item_tuple = (user_item_tuple[0], user_item_tuple[1] - mapper.num_users)
        topk_user_texts = value.get("topk_user_raw_texts", [])[:k]
        topk_item_texts = value.get("topk_item_raw_texts", [])[:k]
        prompt_template = "For the user-item pair, here are some related users and items: Users: {} Items: {}"
        concatenated_texts = prompt_template.format(", ".join(topk_user_texts), ", ".join(topk_item_texts))
        flattened_results[user_item_tuple] = concatenated_texts
    return flattened_results

def merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results):
    # merge the two dictionaries, concat the values only if the key is in both dictionaries
    merged_results = {}
    for key, value in flattened_dense_retrieval_results.items():
        if key in flattened_pagelink_retrieval_results:
            merged_results[key] = value + " \n### " + flattened_pagelink_retrieval_results[key]

    # ablation: only has dense retrieval results
    # merged_results = flattened_dense_retrieval_results
    
    # ablation: only has pagelink retrieval results
    # merged_results = flattened_pagelink_retrieval_results
    return merged_results

def write_to_file(file_name, data):
    """
    Write data to a JSON file, creating directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str + '\n')

def sample_generation(
    args, 
    merged_results, 
    interaction_data, 
    user_proiles,
    item_profiles,
    item_titles,
    explanations,
    pyg_data
):
    count_no_merge = 0
    all_samples = []
    for i, row in tqdm(interaction_data.iterrows(), desc="Generating samples", total=len(interaction_data)):
        user_id = row['user_id']
        item_id = row['item_id']
        user_profile = str(user_proiles.get(user_id, "N/A"))
        item_title = str(item_titles.get(item_id, "N/A"))
        item_profile = str(item_profiles.get(item_id, "N/A"))
        explanation = str(explanations[i])
        
        user_message = "Given the product title, product profile, and user profile, please explain why the user would enjoy this product within 128 words."
        
        item_type = args.dataset.lower()
        user_message += f" {item_type} title: {item_title}. {item_type} profile: {item_profile} User profile: {user_profile}\n### "

        if pyg_data.user_id_to_node.get(user_id) and pyg_data.item_id_to_node.get(item_id):
            ui_pair = tuple((pyg_data.user_id_to_node.get(user_id), pyg_data.item_id_to_node.get(item_id) - len(pyg_data.user_id_to_node)))
        else:
            ui_pair = None

        # ablation: no any retrieval results
        if ui_pair in merged_results:
            user_message += f"{merged_results[ui_pair]}"
        else:
            count_no_merge += 1
        user_message += "\n### Explanation:"    

        user_response = f"### {explanation}"
        # Create sample dictionary
        sample = {
            "user_id": user_id,
            "item_id": item_id,
            "prompt": user_message,
            "chosen": user_response,
            "reject": "I DO NOT KNOW"
        }
        all_samples.append(sample)
    print(f"count_no_merge: {count_no_merge}") 
    return all_samples

def rerank_flattened_results(args, all_samples, mapper, write_file):
    if args.split == "test":
        for sample in all_samples:
            write_to_file(write_file, sample)
        return

    # Initialize the sentence transformer model
    model = TextModel(args.text_encoder)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Process samples in batches
    batch_size = 256
    for i in tqdm(range(0, len(all_samples), batch_size), desc="Reranking samples"):
        batch = all_samples[i:i+batch_size]
        
        user_ids = [sample['user_id'] for sample in batch]
        item_ids = [sample['item_id'] for sample in batch]
        
        user_texts = [mapper.get_user_raw_text(user_id) for user_id in user_ids]
        item_texts = [mapper.get_item_raw_text(item_id) for item_id in item_ids]
        
        combined_texts = [f"{user_text} {item_text}" for user_text, item_text in zip(user_texts, item_texts)]
        
        ground_truths = [sample['chosen'] for sample in batch]
        
        combined_embeddings = model(combined_texts).cpu().numpy()
        ground_truth_embeddings = model(ground_truths).cpu().numpy()

        similarities = cosine_similarity(combined_embeddings, ground_truth_embeddings)
        
        for j, sample in enumerate(batch):
            sample['similarity_score'] = float(similarities[j][j])

    sorted_samples = sorted(all_samples, key=lambda x: x['similarity_score'])

    for sample in sorted_samples:
        write_to_file(write_file, sample)

    print(f"Reranked and wrote {len(sorted_samples)} samples to {write_file}")


def main():
    args = parse_args()

    if not args.saved_model_name:
        args.saved_model_name = f'{args.dataset}_model'
    
    mapper = DataMapper(os.path.join(args.output_dir, f"data_{args.split}.pt"))

    # load dense retrieval results
    with open(os.path.join(args.output_dir, f'dense_retrieval_results_{args.split}.json'), 'r') as f:
        dense_retrieval_results = json.load(f)

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
    interaction_data = pd.read_csv(df_data_path)
    interaction_data['user_id'] = interaction_data['user_id'].map(inv_user_map)
    interaction_data['item_id'] = interaction_data['item_id'].map(inv_item_map)
    print(interaction_data.head(5).to_string())

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
    
    explanations_path = os.path.join(args.dataset_dir, f'{args.split}_explanations.csv')
    explanations_df = pd.read_csv(explanations_path)
    explanations = explanations_df['explanation'].tolist()

    # Define write file
    write_file = os.path.join(args.output_dir, f"translation_{args.split}.json")

    # load pagelink retrieval results
    with open(os.path.join(args.output_dir, f'pagelink_{args.saved_model_name}_{args.split}_pred_edge_to_paths'), 'rb') as f:
        pagelink_retrieval_results = pickle.load(f)

    flattened_dense_retrieval_results = flatten_dense_retrieval_results(dense_retrieval_results, mapper, args.k)
    flattened_pagelink_retrieval_results = flatten_pagelink_retrieval_results(pagelink_retrieval_results, mapper, args.k)

    merged_results = merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results)

    # sample several u-i pairs from the merged results
    # sampled_merged_results = {k: v for i, (k, v) in enumerate(merged_results.items()) if i < 5}
    # print(sampled_merged_results)

    all_samples = sample_generation(
        args, 
        merged_results, 
        interaction_data,
        user_profiles,
        item_profiles,
        item_titles,
        explanations, 
        pyg_data
    )

    # rerank and store the samples
    rerank_flattened_results(args, all_samples, mapper, write_file)

if __name__ == "__main__":
    main()
