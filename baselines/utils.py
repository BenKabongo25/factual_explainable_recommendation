# Ben Kabongo
# July 2025


import ast
import evaluate
import heapq
import json
import math
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from data.utils import build_user_item_map


RMSE = "rmse"
MAE = "mae"
RATING_METRICS = [RMSE, MAE]


BLEU_1 = "bleu1"
BLEU_4 = "bleu4"
METEOR = "meteor"
ROUGE = "rouge"
TEXT_METRICS = [BLEU_1, BLEU_4, METEOR, ROUGE]


def rating_evaluation(
    config: Any, 
    predictions: List[float], 
    references: List[float], 
    metrics: List[str]=RATING_METRICS,
) -> Dict[str, float]:
    results = {}
    
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)

    if RMSE in metrics:
        rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
        results[RMSE] = rmse.item()
        
    if MAE in metrics:
        mae = F.l1_loss(predictions_tensor, actual_ratings)
        results[MAE] = mae.item()
    mae = F.l1_loss(predictions_tensor, actual_ratings)

    return results


def text_evaluation(
    config: Any,
    predictions: List[str], 
    references: List[str],
    metrics: List[str]=TEXT_METRICS
) -> Dict[str, Any]:
    results = {}
    references_list = [[ref] for ref in references]

    if BLEU_1 in metrics or BLEU_4 in metrics:
        bleu_metric = evaluate.load("bleu")

        if BLEU_1 in metrics:
            bleu_results = bleu_metric.compute(predictions=predictions, references=references_list, max_order=1)
            results[BLEU_1] = bleu_results["bleu"]

        if BLEU_4 in metrics:
            bleu_results = bleu_metric.compute(predictions=predictions, references=references_list, max_order=4)
            results[BLEU_4] = bleu_results["bleu"]

    if METEOR in metrics:
        meteor_metric = evaluate.load("meteor")
        meteor_results = meteor_metric.compute(predictions=predictions, references=references)
        results[METEOR] = meteor_results["meteor"]

    if ROUGE in metrics:
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        results[ROUGE + ".1"] = rouge_results["rouge1"]
        results[ROUGE + ".2"] = rouge_results["rouge2"]
        results[ROUGE + ".L"] = rouge_results["rougeL"]
        results[ROUGE + ".Lsum"] = rouge_results["rougeLsum"]

    return results


class WordDictionary:

    def __init__(self) -> None:
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>', '|']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence: str) -> None:
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w: str) -> None:
        if w in ['<bos>', '<eos>', '<pad>', '<unk>', '|']:
            return
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self) -> int:
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size: int = 20000) -> None:
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def ids2tokens(self, ids: List[int]) -> List[str]:
        eos = self.word2idx['<eos>']
        tokens = []
        for i in ids:
            if i == eos:
                break
            tokens.append(self.idx2word[i])
        return tokens
    
    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]


class EntityDictionary:

    def __init__(self) -> None:
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e: str) -> None:
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self) -> int:
        return len(self.idx2entity)
    

class Batchify:

    def __init__(
        self, 
        data: List[Dict[str, Any]],
        word2idx: Dict[str, int],
        seq_len: int = 15, 
        batch_size: int = 128, 
        shuffle: bool = False
    ):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t = [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(x['text'], seq_len, pad, bos, eos))

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self) -> List[torch.Tensor]:
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        return user, item, rating, seq


class DataLoaderFromDataFrameForReviewGeneration:
    
    def __init__(
        self, 
        user_dict: EntityDictionary,
        item_dict: EntityDictionary,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        test_df: pd.DataFrame,
        vocab_size: int = 20000,
        seed: int = 42
    ) -> None:
        self.word_dict = WordDictionary()
        self.user_dict = user_dict
        self.item_dict = item_dict

        self.train_df = train_df
        self.eval_df = eval_df
        self.test_df = test_df

        self.seed = seed
        self.vocab_size = vocab_size

        self.initialize()
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test = self.load_data()

    def initialize(self) -> None:
        for i in range(len(self.train_df)):
            row = self.train_df.iloc[i]
            review = str(row['review'])
            self.word_dict.add_sentence(review)

    def load_data(self) -> List[List[Dict[str, Any]]]:
        data = {'train': [], 'valid': [], 'test': []}
        dfs = {'train': self.train_df, 'valid': self.eval_df, 'test': self.test_df}
        for key in dfs:
            for i in range(len(dfs[key])):
                row = dfs[key].iloc[i]
                user_id = row['user_id']
                item_id = row['item_id']
                rating = row['rating']
                review = str(row['review'])
                data[key].append({'user': self.user_dict.entity2idx[user_id],
                                  'item': self.item_dict.entity2idx[item_id],
                                  'rating': rating,
                                  'text': self.seq2ids(review),
                                  'feature': self.__unk})
                self.feature_set.add('<unk>')

        return data['train'], data['valid'], data['test']

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]
    

class RatingReviewDataset(Dataset):

    def __init__(
        self, 
        config: Any,
        data_df: pd.DataFrame,
        word_dict: WordDictionary,
        user_dict: EntityDictionary, 
        item_dict: EntityDictionary,
    ):
        super().__init__()
        self.config = config
        self.data_df = data_df
        self.word_dict = word_dict
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.reviews = []
        self.reviews_ids = []
        self._reviews2ids()

    def _reviews2ids(self):
        for i in tqdm(range(len(self)), desc="Tokenization", colour="green"):
            review = self.data_df.iloc[i]["review"]
            self.reviews.append(review)

            ids = self.word_dict.tokens2ids(review.split())
            ids = sentence_format(
                ids, 
                self.config.review_length, 
                self.word_dict.word2idx['<pad>'], 
                self.word_dict.word2idx['<bos>'], 
                self.word_dict.word2idx['<eos>']
            )
            self.reviews_ids.append(ids)

    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data_df.iloc[index]
        user_id = self.user_dict.entity2idx[row["user_id"]]
        item_id = self.item_dict.entity2idx[row["item_id"]]
        review = self.reviews[index]
        review_ids = self.reviews_ids[index]

        return {
            "user_id": user_id,
            "item_id": item_id,
            "review": review,
            "review_ids": review_ids
        }
    

def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch

    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def sentence_format(sentence: List[str], max_text_length: int, pad: str, bos: str, eos: str) -> List[str]:
    length = len(sentence)
    if length >= max_text_length:
        return [bos] + sentence[:max_text_length] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_text_length - length)


def ids2tokens(ids: List[int], word2idx: Dict[str, int], idx2word: Dict[int, str]) -> List[str]:
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens


def build_word_dictionnary(data_df: pd.DataFrame) -> Tuple[WordDictionary]:
    word_dict = WordDictionary()
    for i, row in data_df.iterrows():
        review = str(row['review'])
        word_dict.add_sentence(review)
    return word_dict


def postprocess_text(text: str, special_tokens=[]) -> str:
    for token in special_tokens:
        text = text.replace(token, "")
    text = re.sub(r" \'(s|m|ve|d|ll|re)", r"'\1", text)
    text = re.sub(r" \(", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def load_data(config):
    #statement_df = pd.read_csv(
    #    open(os.path.join(config.dataset_dir, "statement_topic_sentiment_freq.csv")),
    #    index_col=0
    #)

    train_df = pd.read_csv(os.path.join(config.dataset_dir, "train_data.csv"))
    eval_df = pd.read_csv(os.path.join(config.dataset_dir, "eval_data.csv"))
    test_df = pd.read_csv(os.path.join(config.dataset_dir, "test_data.csv"))

    if config.use_explanations:
        train_explanations = pd.read_csv(os.path.join(config.dataset_dir, "train_explanations.csv"))
        eval_explanations = pd.read_csv(os.path.join(config.dataset_dir, "eval_explanations.csv"))
        test_explanations = pd.read_csv(os.path.join(config.dataset_dir, "test_explanations.csv"))

        train_df["review"] = train_explanations["explanation"]
        eval_df["review"] = eval_explanations["explanation"]
        test_df["review"] = test_explanations["explanation"]

    #train_df["statement_ids"] = train_df["statement_ids"].apply(ast.literal_eval)
    #eval_df["statement_ids"] = eval_df["statement_ids"].apply(ast.literal_eval)
    #test_df["statement_ids"] = test_df["statement_ids"].apply(ast.literal_eval)

    #id2text = lambda x: " | ".join([statement_df.iloc[idx]["statement"] for idx in x])
    #train_df["review"] = train_df["statement_ids"].apply(id2text)
    #eval_df["review"] = eval_df["statement_ids"].apply(id2text)
    #test_df["review"] = test_df["statement_ids"].apply(id2text)

    user_id2index = json.load(open(os.path.join(config.dataset_dir, "user_id2index.json")))
    item_id2index = json.load(open(os.path.join(config.dataset_dir, "item_id2index.json")))

    return train_df, eval_df, test_df, user_id2index, item_id2index
