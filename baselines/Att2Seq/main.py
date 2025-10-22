import argparse
import ast
import json
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from module import Att2Seq
from baselines.utils import (
    RatingReviewDataset, EntityDictionary,
    build_word_dictionnary, load_data, text_evaluation, set_seed
)


def train(model, config, optimizer, dataloader, loss_fn):
    model.train()
    running_loss = 0.

    for batch in tqdm(dataloader, desc="Training", colour="cyan"):
        U_ids = batch["user_id"]
        I_ids = batch["item_id"]
        T_ids = batch["review_ids"]

        U_ids = torch.LongTensor(U_ids).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(I_ids).to(config.device) # (batch_size,)
        T_ids = torch.stack(T_ids, dim=1).to(dtype=torch.long, device=config.device) # (batch_size, seq_len + 2)

        optimizer.zero_grad()
        logits = model(U_ids, I_ids, T_ids[:, :-1])  # (batch_size,), (batch_size, seq_len + 1, n_tokens)
        loss = loss_fn(logits.view(-1, config.n_tokens), T_ids[:, 1:].reshape((-1,)))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(dataloader)
    return {"loss": running_loss}


def eval(model, config, dataloader, word_dict, metrics=["bleu1", "bleu4", "rouge", "meteor"]):
    model.eval()
    
    references = []
    predictions = []

    with torch.no_grad():    
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Evaluation", colour="cyan", total=len(dataloader)):
            if len(batch) < 2:
                break
            U_ids = batch["user_id"]
            I_ids = batch["item_id"]
            reviews = batch["review"]

            U_ids = torch.LongTensor(U_ids).to(config.device) # (batch_size,)
            I_ids = torch.LongTensor(I_ids).to(config.device)
            reviews_hat = model.generate(U_ids, I_ids, word_dict, config.review_length)

            references.extend(reviews)
            predictions.extend(reviews_hat)

            if config.verbose and batch_idx == 0:
                for i in range(len(reviews)):
                    log = f"User ID: {U_ids[i]}\n"
                    log += f"Item ID: {I_ids[i]}\n"
                    log += f"Review: {reviews[i]}\n"
                    log += f"Generated: {reviews_hat[i]}\n"
                    log += f"-" * 80
                    config.logger.info(log)

    reviews_scores = text_evaluation(config, predictions, references, metrics)
    if config.verbose:
        log = ""
        for metric, score in reviews_scores.items():
            log += f"{metric}: {score:.4f} "
        config.logger.info(log)

    output_df = pd.DataFrame({
        "reference": references,
        "prediction": predictions
    })
    return reviews_scores, output_df


def trainer(model, config, train_dataloader, eval_dataloader, word_dict):
    loss_fn = nn.NLLLoss(ignore_index=config.pad_idx)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)

    train_infos = {}
    eval_infos = {}

    best_bleu = 0.0
    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train(model ,config, optimizer, train_dataloader, loss_fn)
         #train_epoch_infos = {"loss": 0.0}

        for k_1 in train_epoch_infos.keys():
            if k_1 not in train_infos.keys():
                train_infos[k_1] = []
            train_infos[k_1].append(train_epoch_infos[k_1])

        train_loss = train_epoch_infos["loss"]
        desc = f"[{epoch} / {config.n_epochs}] Loss: {train_loss:.4f} best bleu={best_bleu:.4f}"

        if epoch % config.eval_every == 0:
        #if epoch % 1 == 0:
            eval_epoch_infos, _ = eval(model, config, eval_dataloader, word_dict)
            
            for k_1 in eval_epoch_infos.keys():
                if k_1 not in eval_infos.keys():
                    eval_infos[k_1] = []
                eval_infos[k_1].append(eval_epoch_infos[k_1])    

            eval_bleu = eval_epoch_infos["bleu4"]

            if eval_bleu > best_bleu:
                model.save(config.save_model_path)
                best_bleu = eval_bleu

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"test bleu={eval_bleu:.4f} " +
                f"best bleu={best_bleu:.4f}"
            )

        progress_bar.set_description(desc)
        config.logger.info(desc)

        results = {"train": train_infos, "eval": eval_infos}
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, eval_infos


def get_data(config):
    (train_df, eval_df, test_df, user_id2index, item_id2index) = load_data(config)

    user_dict = EntityDictionary()
    user_dict.entity2idx = user_id2index
    user_dict.idx2entity = {v: k for k, v in user_id2index.items()}

    item_dict = EntityDictionary()
    item_dict.entity2idx = item_id2index
    item_dict.idx2entity = {v: k for k, v in item_id2index.items()}

    word_dict = build_word_dictionnary(train_df)
    word_dict.keep_most_frequent(config.vocab_size)

    train_dataset = RatingReviewDataset(config, train_df, word_dict, user_dict, item_dict)
    eval_dataset = RatingReviewDataset(config, eval_df, word_dict, user_dict, item_dict)
    test_dataset = RatingReviewDataset(config, test_df, word_dict, user_dict, item_dict)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return (word_dict, user_dict, item_dict), (train_dataloader, eval_dataloader, test_dataloader)


def main(config):
    set_seed(config.seed)

    model_name = "Att2Seq"
    task_type = "review" if not config.use_explanations else "explanation"
    config.save_dir = os.path.join(config.save_dir, f"{model_name}_{task_type}", config.dataset_name)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("Att2Seq" + config.dataset_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    config.logger = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    (word_dict, user_dict, item_dict), (train_dataloader, eval_dataloader, test_dataloader) = get_data(config)

    config.n_users = len(user_dict)
    config.n_items = len(item_dict)
    config.n_tokens = len(word_dict)
    config.bos_idx = word_dict.word2idx['<bos>']
    config.pad_idx = word_dict.word2idx['<pad>']
    
    model = Att2Seq(
        config.n_users, 
        config.n_items, 
        config.n_tokens, 
        config.embedding_dim, 
        config.hidden_size, 
        config.dropout, 
        config.n_layers
    )
    if config.load_model:
        model.load_model(config.save_model_path)
    model = model.to(device)

    if config.verbose:
        log = "Att2Seq\n\n"
        for k, v in vars(config).items():
            log += f"{k}: {v}\n"
        log += f"{'-' * 80}\n\n"
        for batch in train_dataloader:
            for keys in batch.keys():
                batch[keys] = batch[keys][:5]
        log += f"{batch}\n"
        config.logger.info("\n" + log)

    train_infos, eval_infos = trainer(model, config, train_dataloader, eval_dataloader, word_dict)
    #train_infos, eval_infos = {}, {}
    model.load(config.save_model_path)
    model = model.to(device)
    test_infos, output_df = eval(
        model, config, test_dataloader, word_dict,
        metrics=["bleu1", "bleu4", "rouge"]
    )
    output_df.to_csv(os.path.join(config.save_dir, "output.csv"), index=False)

    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    config.logger.info(f"Results: {results}")
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Att2Seq (EACL\'17) without rating input')

    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='size of user/item embeddings')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='number of hidden units')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--review_length', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='keep the most frequent words in the vocabulary')

    parser.add_argument('--n_epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    parser.add_argument("--use_explanations", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use explanations for training.")

    config = parser.parse_args()
    main(config)
