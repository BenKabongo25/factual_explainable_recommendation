# Ben Kabongo
# September 2025


import argparse
import json
import math
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

from data.utils import build_user_item_map, Constants


def temporal_train_eval_test_split(
    data_df: pd.DataFrame,
    train_size: float = 0.8,
    eval_size: float = 0.1,
    test_size: float = 0.1,
    columns : List[str] = Constants.COLUMNS,
    min_interactions: int = 5,
    delete_cold_start_items: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and test sets based on temporal order for each user.

    Args:
        data_df (pd.DataFrame): DataFrame containing the dataset with columns:
        train_size (float): Proportion of data to use for training.
        eval_size (float): Proportion of data to use for validation.
        test_size (float): Proportion of data to use for testing.
        columns (List[str]): List of columns to retain in the DataFrame.
        min_interactions (int): Minimum number of interactions a user must have to be included in the split.
        delete_cold_start_items (bool): If True, removes items in eval/test sets that are not in the training set.
    Returns:
        Tuple containing:
            - train_df (pd.DataFrame): DataFrame for training set.
            - eval_df (pd.DataFrame): DataFrame for validation set.
            - test_df (pd.DataFrame): DataFrame for test set.
    """
    data_df = data_df[columns]
    data_df["user_id"] = data_df["user_id"].astype(str)
    data_df["item_id"] = data_df["item_id"].astype(str)
    data_df = data_df.dropna(subset=columns)
    data_df = data_df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    total_size = train_size + eval_size + test_size
    if total_size != 1.0:
        train_size = train_size / total_size
        eval_size = eval_size / total_size
        test_size = test_size / total_size

    train_splits = []
    eval_splits = []
    test_splits = []

    for _, group in data_df.groupby("user_id"):
        n = len(group)
        if n < min_interactions:
            continue

        n_train = int(math.floor(n * train_size))
        n_eval = int(math.floor(n * (train_size + eval_size)))

        if n - n_eval == 0:
            n_train = n - 2
            n_eval = n - 1 
    
        train_splits.append(group.iloc[:n_train])
        eval_splits.append(group.iloc[n_train:n_eval])
        test_splits.append(group.iloc[n_eval:])

    train_df = pd.concat(train_splits).reset_index(drop=True)
    eval_df = pd.concat(eval_splits).reset_index(drop=True)
    test_df = pd.concat(test_splits).reset_index(drop=True)

    if delete_cold_start_items:
        train_items = set(train_df["item_id"].unique())
        eval_df = eval_df[eval_df["item_id"].isin(train_items)].reset_index(drop=True)
        test_df = test_df[test_df["item_id"].isin(train_items)].reset_index(drop=True)

    return train_df, eval_df, test_df


def user_train_eval_test_split(
    data_df: pd.DataFrame,
    train_size: float = 0.8,
    eval_size: float = 0.1,
    test_size: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits users into training, validation, and test sets.

    Args:
        data_df (pd.DataFrame): DataFrame containing the dataset with a "user_id" column.
        train_size (float): Proportion of users to use for training.
        eval_size (float): Proportion of users to use for validation.
        test_size (float): Proportion of users to use for testing.
    Returns:
        Tuple containing:
            - train_users (List[str]): List of user IDs for the training set.
            - eval_users (List[str]): List of user IDs for the validation set.
            - test_users (List[str]): List of user IDs for the test set.
    """
    users = data_df["user_id"].unique().tolist()
    n_users = len(users)

    n_eval = int(math.floor(n_users * eval_size))
    eval_users = users[:n_eval]
    n_test = int(math.floor(n_users * test_size))
    test_users = users[n_eval:n_eval + n_test]
    train_users = users[n_eval + n_test:]

    return train_users, eval_users, test_users


def user_train_eval_test_split_stratified(
    data_df: pd.DataFrame,
    train_size: float = 0.8,
    eval_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    qbins : List[float] =(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Splits users into training, validation, and test sets in a stratified manner based on user activity.

    Args:
        data_df (pd.DataFrame): DataFrame containing the dataset with a "user_id" column.
        train_size (float): Proportion of users to use for training.
        eval_size (float): Proportion of users to use for validation.
        test_size (float): Proportion of users to use for testing.
        seed (int): Random seed for reproducibility.
        qbins (List[float]): Quantile bins for stratification based on user activity.
    Returns:
        Tuple containing:
            - train_users (List[str]): List of user IDs for the training set.
            - eval_users (List[str]): List of user IDs for the validation set.
            - test_users (List[str]): List of user IDs for the test set.
    """
    user_counts = data_df.groupby("user_id")["item_id"].size().rename("n_interactions")
    users_df = user_counts.to_frame().reset_index()

    users_df["act_bin"] = pd.qcut(
        users_df["n_interactions"],
        q=qbins,
        duplicates="drop",
        labels=False
    )

    rng = np.random.default_rng(seed)
    train_users, eval_users, test_users = [], [], []

    for _, grp in users_df.groupby("act_bin", dropna=False):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_eval = int(math.floor(n * eval_size))
        n_test = int(math.floor(n * test_size))
        n_train = n - n_eval - n_test

        sel_train = grp.loc[idx[:n_train], "user_id"].tolist()
        sel_eval  = grp.loc[idx[n_train:n_train+n_eval], "user_id"].tolist()
        sel_test  = grp.loc[idx[n_train+n_eval:n_train+n_eval+n_test], "user_id"].tolist()

        train_users.extend(sel_train)
        eval_users.extend(sel_eval)
        test_users.extend(sel_test)

    return train_users, eval_users, test_users



def main(config):
    processed_data_df = pd.read_csv(os.path.join(config.dataset_dir, config.processed_data_file))

    print("Splitting dataset into train, validation, and test sets...")
    train_df, eval_df, test_df = temporal_train_eval_test_split(
        processed_data_df,
        train_size=config.time_train_size,
        eval_size=config.time_eval_size,
        test_size=config.time_test_size,
        min_interactions=config.min_interactions,
        delete_cold_start_items=config.delete_cold_start_items
    )
    train_df.to_csv(os.path.join(config.dataset_dir, "train_data.csv"), index=False)
    eval_df.to_csv(os.path.join(config.dataset_dir, "eval_data.csv"), index=False)
    test_df.to_csv(os.path.join(config.dataset_dir, "test_data.csv"), index=False)
    print("Dataset split completed.")

    print("Building user and item mappings...")
    user_id2index, item_id2index = build_user_item_map(
        pd.concat([train_df, eval_df, test_df], ignore_index=True)
    )
    with open(os.path.join(config.dataset_dir, "user_id2index.json"), "w") as f:
        json.dump(user_id2index, f)
    with open(os.path.join(config.dataset_dir, "item_id2index.json"), "w") as f:
        json.dump(item_id2index, f)
    print("User and item mappings completed.")

    print("Splitting users into train, validation, and test sets...")
    train_users, eval_users, test_users = user_train_eval_test_split_stratified(
        processed_data_df,
        train_size=config.user_train_size,
        eval_size=config.user_eval_size,
        test_size=config.user_test_size,
        seed=config.seed
    )
    with open(os.path.join(config.dataset_dir, "users_train.json"), "w") as f:
        json.dump(train_users, f)
    with open(os.path.join(config.dataset_dir, "users_eval.json"), "w") as f:
        json.dump(eval_users, f)
    with open(os.path.join(config.dataset_dir, "users_test.json"), "w") as f:
        json.dump(test_users, f)
    print("Users split completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--processed_data_file", type=str, default="processed_dataset.csv")

    # temporal split args
    parser.add_argument("--time_train_size", type=float, default=0.8)
    parser.add_argument("--time_eval_size", type=float, default=0.1)
    parser.add_argument("--time_test_size", type=float, default=0.1)
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--delete_cold_start_items", action=argparse.BooleanOptionalAction, default=True)

    # user split args
    parser.add_argument("--user_train_size", type=float, default=0.8)
    parser.add_argument("--user_eval_size", type=float, default=0.1)
    parser.add_argument("--user_test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    config = parser.parse_args()
    main(config)