import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import Dataset, DataLoader
from utils.parse import args


class Dataset(Dataset):
    def __init__(self, user_list, item_list):
        self.user_list = user_list
        self.item_list = item_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return self.user_list[index], self.item_list[index]


class TripleData(Dataset):
    def __init__(self, user_list, pos_item_list, neg_item_list):
        self.user_list = user_list
        self.pos_item_list = pos_item_list
        self.neg_item_list = neg_item_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return (
            self.user_list[index],
            self.pos_item_list[index],
            self.neg_item_list[index],
        )


class DataHandler:

    def __init__(
        self,
        dataset_dir,
        user_id_col="user_id",
        item_id_col="item_id",
        train_file="train_data.csv",
        val_file="eval_data.csv",
        test_file="test_data.csv",
    ):
        self.dataset_dir = dataset_dir
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

        self.trn_path = os.path.join(self.dataset_dir, train_file)
        self.val_path = os.path.join(self.dataset_dir, val_file)
        self.tst_path = os.path.join(self.dataset_dir, test_file)

        self._process()

    def _process(self):
        self.train_df = pd.read_csv(self.trn_path)
        self.eval_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.tst_path)

        data_df = pd.concat([self.train_df, self.eval_df, self.test_df], ignore_index=True)
        self.all_users = data_df[self.user_id_col].unique().tolist()
        self.all_items = data_df[self.item_id_col].unique().tolist()

        self.user_map = {uid: u for uid, u in enumerate(self.all_users)}
        self.item_map = {iid: i for iid, i in enumerate(self.all_items)}

        inv_user_map = {u: uid for uid, u in self.user_map.items()}
        inv_item_map = {i: iid for iid, i in self.item_map.items()}

        self.train_df[self.user_id_col] = self.train_df[self.user_id_col].map(inv_user_map)
        self.train_df[self.item_id_col] = self.train_df[self.item_id_col].map(inv_item_map)
        self.eval_df[self.user_id_col] = self.eval_df[self.user_id_col].map(inv_user_map)
        self.eval_df[self.item_id_col] = self.eval_df[self.item_id_col].map(inv_item_map)
        self.test_df[self.user_id_col] = self.test_df[self.user_id_col].map(inv_user_map)
        self.test_df[self.item_id_col] = self.test_df[self.item_id_col].map(inv_item_map)

        self.user_num = len(self.all_users)
        self.item_num = len(self.all_items)

        self.user_num_val = self.eval_df[self.user_id_col].nunique()
        self.user_num_tst = self.test_df[self.user_id_col].nunique()
        self.item_num_val = self.eval_df[self.item_id_col].nunique()
        self.item_num_tst = self.test_df[self.item_id_col].nunique()

    def _build_ground_true_implicit(self, data_df):
        gt = {user: set() for user in self.user_map}
        for u_id, i_id in zip(data_df[self.user_id_col], data_df[self.item_id_col]):
            gt[u_id].add(i_id)
        return gt
    
    def get_eval_ground_truth(self):
        return self._build_ground_true_implicit(self.eval_df)
    
    def _get_split(self, split):
        if split == "train":
            return self.train_df
        elif split == "val":
            return self.eval_df
        elif split == "test":
            return self.test_df
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

    def load_csv(self, split):
        """return user, item as list"""
        df = self._get_split(split)
        user_list = df[self.user_id_col].tolist()
        item_list = df[self.item_id_col].tolist()
        return user_list, item_list

    def load_csv_with_negative_sampling(self, split):
        """return user, item as list"""
        df = self._get_split(split)
        user_list = df[self.user_id_col].tolist()
        pos_item_list = df[self.item_id_col].tolist()

        all_items = df[self.item_id_col].unique()
        user_interacted_items = df.groupby(self.user_id_col)[self.item_id_col].apply(set).to_dict()

        neg_item_list = []
        for index, row in df.iterrows():
            user = row[self.user_id_col]
            user_items = user_interacted_items[user]
            negative_item = random.choice(all_items)
            while negative_item in user_items:
                negative_item = random.choice(all_items)
            neg_item_list.append(negative_item)
        return user_list, pos_item_list, neg_item_list

    def create_adjacency_matrix(self, split="train"):
        user_list, item_list = self.load_csv(split)
        # Create coo matrix
        adj_matrix = coo_matrix(
            (np.ones(len(user_list)), (user_list, item_list)),
            shape=(self.user_num, self.item_num),
        )
        return adj_matrix

    def make_torch_adj(self, adj_matrix):
        a = csr_matrix((self.user_num, self.user_num))
        b = csr_matrix((self.item_num, self.item_num))

        mat = sp.vstack(
            [sp.hstack([a, adj_matrix]), sp.hstack([adj_matrix.transpose(), b])]
        )
        mat = (mat != 0) * 1.0

        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)

    def load_data(self):
        # load training triple batch
        user_list, pos_item_list, neg_item_list = self.load_csv_with_negative_sampling("train")
        trn_dataset = TripleData(user_list, pos_item_list, neg_item_list)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        user_list, item_list = self.load_csv("val")
        val_dataset = Dataset(user_list, item_list)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        user_list, item_list = self.load_csv("test")
        tst_dataset = Dataset(user_list, item_list)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True)

        return trn_loader, val_loader, tst_loader

    def load_mat(self):
        trn_mat = self.create_adjacency_matrix("train")
        val_mat = self.create_adjacency_matrix("val")
        tst_mat = self.create_adjacency_matrix("test")

        trn_mat = self.make_torch_adj(trn_mat)
        val_mat = self.make_torch_adj(val_mat)
        tst_mat = self.make_torch_adj(tst_mat)
        return trn_mat, val_mat, tst_mat
