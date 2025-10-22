# Ben Kabongo
# July 2025

# PEPLER for review generation
# https://dl.acm.org/doi/pdf/10.1145/3580488


import argparse
import ast
import json
import logging
import math
import pandas as pd
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from module import PEPLER
from dataset import RatingReviewDataset
from baselines.utils import (
    load_data, rating_evaluation, text_evaluation,
    postprocess_text, set_seed
)


def empty_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()


def train(model, config, optimizer, rating_criterion, dataloader):
    model.train()
    text_loss = 0.
    rating_loss = 0.

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Training", colour="green", total=len(dataloader))
    for batch_idx, batch in progress_bar:
        empty_cache()
        user = torch.LongTensor(batch['user_id']).to(config.device)
        item = torch.LongTensor(batch['item_id']).to(config.device)
        rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
        seq = torch.LongTensor(batch['tokens']).to(config.device)
        mask = torch.LongTensor(batch['mask']).to(config.device)
        
        optimizer.zero_grad()
        outputs, rating_p = model(user, item, seq, mask)
        t_loss = outputs.loss
        r_loss = rating_criterion(rating_p, rating)
        loss = config.text_reg * t_loss + config.rating_reg * r_loss
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()

        if batch_idx % config.log_interval == 0 or batch_idx == len(dataloader):
            cur_t_loss = text_loss / len(dataloader)
            cur_r_loss = rating_loss / len(dataloader)
            description = 'text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, batch_idx, len(dataloader))
            logging.info(description)
            progress_bar.set_description(description)
            text_loss = 0.
            rating_loss = 0.

    text_loss /= len(dataloader)
    rating_loss /= len(dataloader)
    return {'text_loss': text_loss, 'rating_loss': rating_loss}


def evaluate(model, config, rating_criterion, dataloader):
    model.eval()
    text_loss = 0.
    rating_loss = 0.

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Eval", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['tokens']).to(config.device)
            mask = torch.LongTensor(batch['mask']).to(config.device)

            outputs, rating_p = model(user, item, seq, mask)
            t_loss = outputs.loss
            r_loss = rating_criterion(rating_p, rating)

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
    
    text_loss /= len(dataloader)
    rating_loss /= len(dataloader)
    return {'text_loss': text_loss, 'rating_loss': rating_loss}


def rating_and_evaluate(model, config, dataloader):
    model.eval()
    users = []
    items = []
    reference_ratings = []
    predict_ratings = []

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Rating eval", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['tokens']).to(config.device)
            mask = torch.LongTensor(batch['mask']).to(config.device)
            _, rating_p = model(user, item, seq, mask)

            reference_ratings.extend(rating.cpu().detach().numpy().tolist())
            predict_ratings.extend(rating_p.cpu().detach().numpy().tolist())
            users.extend(user.cpu().detach().numpy().tolist())
            items.extend(item.cpu().detach().numpy().tolist())

    ratings_scores = rating_evaluation(config, predictions=predict_ratings, references=reference_ratings)
    output_df = pd.DataFrame({
        'user_id': users, 
        'item_id': items, 
        'reference': reference_ratings,
        'prediction': predict_ratings
    })
    return {'rating': ratings_scores, 'output': output_df}


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    text = " ".join(tokens)
    text = postprocess_text(text, [eos])
    return text


def generate_and_evaluate(model, config, tokenizer, dataloader):
    model.eval()

    users = []
    items = []
    reference_texts = []
    reference_ratings = []
    predict_texts = []
    predict_ratings = []

    progress_bar = tqdm(enumerate(dataloader, 1), desc="Generate", colour="green", total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            empty_cache()
            user = torch.LongTensor(batch['user_id']).to(config.device)
            item = torch.LongTensor(batch['item_id']).to(config.device)
            rating = torch.tensor(batch['rating'].clone().detach(), dtype=torch.float32).to(config.device)
            seq = torch.LongTensor(batch['tokens']).to(config.device)
            review = batch['review']

            users.extend(user.cpu().detach().numpy().tolist())
            items.extend(item.cpu().detach().numpy().tolist())
            reference_texts.extend(review)
            reference_ratings.extend(rating.cpu().detach().numpy().tolist())

            text = seq[:, :1].to(config.device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                if idx == 0:
                    outputs, rating_p = model(user, item, text, None)
                    predict_ratings.extend(rating_p.cpu().detach().numpy().tolist())
                else:
                    outputs, _ = model(user, item, text, None, False)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            review_p = [ids2tokens(rids, tokenizer, config.eos) for rids in ids]
            predict_texts.extend(review_p)

    ratings_scores = rating_evaluation(config, predictions=predict_ratings, references=reference_ratings)
    review_scores = text_evaluation(config, predictions=predict_texts, references=reference_texts)
    output_df = pd.DataFrame({
        'reference': reference_texts,
        'prediction': predict_texts,
    })
    return {'text': review_scores, 'rating': ratings_scores, 'output': output_df}


def trainer(model, config, optimizer, rating_criterion, train_dataloader, val_dataloader):
    best_val_loss = float('inf')
    endure_count = 0

    train_infos = {"text_loss": [], "rating_loss": []}
    val_infos = {"text_loss": [], "rating_loss": []}

    for epoch in range(1, config.n_epochs + 1):
        logging.info('epoch {}'.format(epoch))

        train_epoch_infos = train(model, config, optimizer, rating_criterion, train_dataloader)
        val_epoch_infos = evaluate(model, config, rating_criterion, val_dataloader)

        train_infos['text_loss'].append(train_epoch_infos['text_loss'])
        train_infos['rating_loss'].append(train_epoch_infos['rating_loss'])
        val_infos['text_loss'].append(val_epoch_infos['text_loss'])
        val_infos['rating_loss'].append(val_epoch_infos['rating_loss'])

        val_loss = val_epoch_infos['text_loss'] + val_epoch_infos['rating_loss']

        logging.info('text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
            math.exp(val_epoch_infos['text_loss']), val_epoch_infos['rating_loss'], val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.model_path)
        else:
            endure_count += 1
            logging.info('Endured {} time(s)'.format(endure_count))
            if endure_count == config.endure_times:
                logging.info('Cannot endure it anymore | Exiting from early stop')
                break

    return {'train': train_infos, 'val': val_infos}


def main(config):
    set_seed(config.seed)

    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "PEPLER"
    task_type = "review" if not config.use_explanations else "explanation"
    config.save_dir = os.path.join(config.save_dir, f"{model_name}_{task_type}", config.dataset_name)

    os.makedirs(config.save_dir, exist_ok=True)
    config.model_path = os.path.join(config.save_dir, 'model.pth')
    config.results_path = os.path.join(config.save_dir, 'results.json')

    logging.basicConfig(filename=os.path.join(config.save_dir, 'log.txt'), level=logging.INFO)

    # Load data
    logging.info('Loading data')
    config.bos = '<bos>'
    config.eos = '<eos>'
    config.pad = '<pad>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=config.bos, eos_token=config.eos, pad_token=config.pad)

    train_df, eval_df, test_df, user_id2index, item_id2index = load_data(config)

    train_size, test_size, eval_size = len(train_df), len(test_df), len(eval_df)
    logging.info("Data shape: {}".format(train_df.shape))
    logging.info("Train size: {}, Test size: {}, Validation size: {}".format(train_size, test_size, eval_size))
    logging.info("{}".format(train_df.head()))

    config.n_users = len(user_id2index)
    config.n_items = len(item_id2index)

    train_df['user_id'] = train_df['user_id'].apply(lambda u: user_id2index[u])
    train_df['item_id'] = train_df['item_id'].apply(lambda i: item_id2index[i])
    test_df['user_id'] = test_df['user_id'].apply(lambda u: user_id2index[u])
    test_df['item_id'] = test_df['item_id'].apply(lambda i: item_id2index[i])
    eval_df['user_id'] = eval_df['user_id'].apply(lambda u: user_id2index[u])
    eval_df['item_id'] = eval_df['item_id'].apply(lambda i: item_id2index[i])
    eval_df['rating'] = eval_df['rating'].astype(float)

    train_dataset = RatingReviewDataset(config, train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = RatingReviewDataset(config, test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    eval_dataset = RatingReviewDataset(config, eval_df, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    # Build the model
    config.n_tokens = len(tokenizer)
    model = PEPLER.from_pretrained('gpt2', config.n_users, config.n_items, config.use_mf)
    model.resize_token_embeddings(config.n_tokens)  # three tokens added, update embedding table
    if config.load_model:
        model.load(config.model_path)
    model = model.to(config.device)

    rating_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Train the model
    logging.info('Start training')
    results = trainer(model, config, optimizer, rating_criterion, train_dataloader, eval_dataloader)
    #results = {}

    # Test
    model.load(config.model_path)
    model = model.to(config.device)
    logging.info('Start testing')
    test_results = generate_and_evaluate(model, config, tokenizer, test_dataloader)
    results['test'] = {'text': test_results['text'], 'rating': test_results['rating']}
    output_df = test_results['output']

    # Save results
    logging.info('Saving results')
    output_df.to_csv(os.path.join(config.save_dir, 'output.csv'), index=False)
    with open(config.results_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')

    parser.add_argument('--dataset_name', type=str, default=None,
                        help='name of the dataset, e.g., "beauty"')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./PEPLER/',
                        help='directory to save the final model')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)
    
    parser.add_argument('--threshold_rating', type=float, default=3.0,
                        help='threshold for rating')
    parser.add_argument('--ranking_metrics_flag', action=argparse.BooleanOptionalAction, default=True,
                        help='whether to compute ranking metrics')
    parser.add_argument('--lang', type=str, default='en',
                        help='language')

    parser.add_argument('--endure_times', type=int, default=100,
                        help='the maximum endure times of loss increasing on validation')
    parser.add_argument('--rating_reg', type=float, default=0.01,
                        help='regularization on recommendation task')
    parser.add_argument('--text_reg', type=float, default=1.0,
                        help='regularization on text generation task')
    parser.add_argument('--use_mf', action=argparse.BooleanOptionalAction, default=False,
                        help='otherwise MLP')
    parser.add_argument('--review_length', type=int, default=128,
                        help='number of words to generate for each sample')

    parser.add_argument("--use_explanations", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use explanations for training.")
    
    config = parser.parse_args()
    main(config)
