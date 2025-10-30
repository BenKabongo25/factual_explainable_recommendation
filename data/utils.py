# Ben Kabongo
# September 2025


import ast
import pandas as pd
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union


class Constants:
    """Class to hold constant values used across the project."""
    
    SPLITS = ["train", "eval", "test"]
    COLUMNS = ["user_id", "item_id", "timestamp", "rating", "statement_ids", "topic_ids", "sentiments", "review"]


def build_user_item_map(
    data_df: pd.DataFrame
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build mappings from user IDs and item IDs to unique integer indices.

    Args:
        data_df (pd.DataFrame): DataFrame containing user-item interactions.
    Returns:
        Tuple containing:
            - user_map (Dict[str, int]): Mapping from user IDs to indices.
            - item_map (Dict[str, int]): Mapping from item IDs to indices.
    """
    data_df["user_id"] = data_df["user_id"].astype(str)
    data_df["item_id"] = data_df["item_id"].astype(str)

    user_map = {user: idx for idx, user in enumerate(data_df["user_id"].unique())}
    item_map = {item: idx for idx, item in enumerate(data_df["item_id"].unique())}

    return user_map, item_map


def ranking_to_pair_dataset(
    ranking_df: pd.DataFrame,
    selected_users: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert a ranking dataset to a pairwise dataset.

    Args:
        ranking_df (pd.DataFrame): DataFrame containing ranking data with columns
            "user_id", "positive_item_id", and "negative_item_ids" (as a string
            representation of a list).
        selected_users (Optional[List[str]]): List of user IDs to include in the
            output dataset. If None, all users are included.
    Returns:
        pd.DataFrame: DataFrame containing pairwise data with columns "user_id",
            "item_id", and "label" (1 for positive items, 0 for negative items).
    """
    pair_data = {
        "user_id": [],
        "item_id": [],
        "label": []
    }

    for _, row in tqdm(ranking_df.iterrows(), total=len(ranking_df), desc="Converting rankings to pairs"):
        user_id = str(row["user_id"])
        if selected_users is not None and user_id not in selected_users:
            continue

        positive_item_id = str(row["positive_item_id"])
        pair_data["user_id"].append(user_id)
        pair_data["item_id"].append(positive_item_id)
        pair_data["label"].append(1)

        negative_item_ids = ast.literal_eval(row["negative_item_ids"])
        for rank, item_id in enumerate(negative_item_ids):
            pair_data["user_id"].append(user_id)
            pair_data["item_id"].append(str(item_id))
            pair_data["label"].append(0)

    pair_df = pd.DataFrame(pair_data)
    return pair_df


def get_similar_ids(
    similarity_df: pd.DataFrame,
    target_id: int,
    accepted_ids: Optional[Union[List[int], torch.LongTensor]] = None,
    threshold: float = 0.8
) -> Tuple[List[int], List[int]]:
    """
    Retrieve similar IDs for a given target ID from a similarity DataFrame.

    Args:
        similarity_df (pd.DataFrame): DataFrame containing similarity data with
            columns "index", "similars" (list of similar IDs), and "similarities" (list of similarity scores).
        target_id (int): The target ID for which to find similar IDs.
        accepted_ids (Optional[Union[List[int], torch.LongTensor]]): List or
            tensor of accepted IDs to filter the similar IDs. If None, no filtering
            is applied.
        threshold (float): Minimum similarity score to consider a similar ID.

    Returns:
        Tuple containing:
            - List[int]: List of similar IDs for the target ID, filtered by accepted_ids
                if provided.
            - List[int]: Corresponding similarity scores for the similar IDs.
    """
    sub_df = similarity_df[similarity_df["index"] == target_id]
    if sub_df.empty:
        return [], []
    
    if accepted_ids is not None:
        if isinstance(accepted_ids, torch.LongTensor):
            accepted_ids = accepted_ids.tolist()

    similars = sub_df.iloc[0]["similars"]
    similarities = sub_df.iloc[0]["similarities"]
    remained_similars = []
    remained_similarities = []
    for sim_id, sim_score in zip(similars, similarities):
        if sim_score >= threshold and sim_id in accepted_ids:
            remained_similars.append(sim_id)
            remained_similarities.append(sim_score)

    return remained_similars, remained_similarities