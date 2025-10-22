import os
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List


class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(
        self, 
        domain,
        dataset_dir, 
        output_dir="./output",
        profiles_dir="./profiles",
        user_id_col="user_id",
        item_id_col="item_id",
        train_file="train_data.csv",
        val_file="eval_data.csv",
        test_file="test_data.csv",
        user_profile_file="user_summaries.csv",
        item_profile_file="item_summaries.csv",
        profile_col="summary_text",
        train_explanation_file="train_explanations.csv",
        val_explanation_file="eval_explanations.csv",
        test_explanation_file="test_explanations.csv",
        explanation_col="explanation",
        item_title_col="title",
        item_description_col="description"
    ):
        self.system_prompt = "Explain why the user would interact with the item."
        self.domain = domain.lower()

        self.dataset_dir = dataset_dir
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.profile_col = profile_col
        self.explanation_col = explanation_col
        self.item_title_col = item_title_col
        self.item_description_col = item_description_col

        self.trn_path = os.path.join(self.dataset_dir, train_file)
        self.val_path = os.path.join(self.dataset_dir, val_file)
        self.tst_path = os.path.join(self.dataset_dir, test_file)

        self.user_profile_path = os.path.join(profiles_dir, user_profile_file)
        self.item_profile_path = os.path.join(profiles_dir, item_profile_file)

        self.trn_explanation_path = os.path.join(self.dataset_dir, train_explanation_file)
        self.val_explanation_path = os.path.join(self.dataset_dir, val_explanation_file)
        self.tst_explanation_path = os.path.join(self.dataset_dir, test_explanation_file)

        user_path = os.path.join(output_dir, "user_emb.pkl")
        item_path = os.path.join(output_dir, "item_emb.pkl")
        with open(user_path, "rb") as file:
            self.user_emb = pickle.load(file)
        with open(item_path, "rb") as file:
            self.item_emb = pickle.load(file)

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

        for df in [self.train_df, self.eval_df, self.test_df]:
            df[self.user_id_col] = df[self.user_id_col].map(inv_user_map)
            df[self.item_id_col] = df[self.item_id_col].map(inv_item_map)

        self.user_num = len(self.all_users)
        self.item_num = len(self.all_items)

        self.user_num_val = self.eval_df[self.user_id_col].nunique()
        self.user_num_tst = self.test_df[self.user_id_col].nunique()
        self.item_num_val = self.eval_df[self.item_id_col].nunique()
        self.item_num_tst = self.test_df[self.item_id_col].nunique()

        user_profiles = pd.read_csv(self.user_profile_path)
        user_profiles[self.user_id_col] = user_profiles[self.user_id_col].map(inv_user_map)
        self.user_profiles = user_profiles.set_index(self.user_id_col).to_dict()[self.profile_col]

        item_infos = pd.read_csv(self.item_profile_path)
        item_infos[self.item_id_col] = item_infos[self.item_id_col].map(inv_item_map)
        item_infos = item_infos.set_index(self.item_id_col).to_dict()

        self.item_titles = item_infos[self.item_title_col]
        self.item_descriptions = item_infos[self.item_description_col]
        self.item_profiles = item_infos[self.profile_col]

        self.trn_explanations = pd.read_csv(self.trn_explanation_path)
        self.val_explanations = pd.read_csv(self.val_explanation_path)
        self.tst_explanations = pd.read_csv(self.tst_explanation_path)
        
    def _build_train_input(self):
        inputs = []
        for i in range(len(self.train_df)):
            user_id = self.train_df.iloc[i][self.user_id_col]
            item_id = self.train_df.iloc[i][self.item_id_col]

            item_title = str(self.item_titles.get(item_id, self.domain))
            item_profile = str(self.item_profiles.get(item_id, self.item_descriptions.get(item_id, "")))
            user_profile = str(self.user_profiles.get(user_id, ""))
            explanation = str(self.trn_explanations.iloc[i][self.explanation_col])

            user_embed = self.user_emb[user_id]
            item_embed = self.item_emb[item_id]

            user_message = f"user record: <USER_EMBED> {self.domain} record: <ITEM_EMBED> {self.domain} name: {item_title} user profile: {user_profile} {self.domain} profile: {item_profile} <EXPLAIN_POS> {explanation}"
            inputs.append(
                (
                    user_embed,
                    item_embed,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                )
            )
        return inputs
    
    def _build_split_input(self, df, explanation_df):
        inputs = []
        for i in range(len(df)):
            user_id = df.iloc[i][self.user_id_col]
            item_id = df.iloc[i][self.item_id_col]

            item_title = str(self.item_titles.get(item_id, self.domain))
            item_profile = str(self.item_profiles.get(item_id, self.item_descriptions.get(item_id, "")))
            user_profile = str(self.user_profiles.get(user_id, ""))
            explanation = str(explanation_df.iloc[i][self.explanation_col])

            user_embed = self.user_emb[user_id]
            item_embed = self.item_emb[item_id]

            user_message = f"user record: <USER_EMBED> {self.domain} record: <ITEM_EMBED> {self.domain} name: {item_title} user profile: {user_profile} {self.domain} profile: {item_profile} <EXPLAIN_POS>"
            inputs.append(
                (
                    user_embed,
                    item_embed,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    explanation,
                )
            )
        return inputs
        
    def load_data(self):
        trn_input = self._build_train_input()
        val_input = self._build_split_input(self.eval_df, self.val_explanations)
        tst_input = self._build_split_input(self.test_df, self.tst_explanations)

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False)

        return trn_loader, val_loader, tst_loader
