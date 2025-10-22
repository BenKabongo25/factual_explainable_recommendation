import argparse
import ast
import json
import logging
import math
import os
import pandas as pd
import torch
import torch.nn as nn

from module import NRT
from baselines.utils import (
    Batchify, DataLoaderFromDataFrameForReviewGeneration, EntityDictionary, 
    RATING_METRICS, TEXT_METRICS,
    build_word_dictionnary, ids2tokens, load_data, postprocess_text, rating_evaluation, set_seed, text_evaluation
)


parser = argparse.ArgumentParser(description='NRT')
parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
parser.add_argument('--dataset_dir', type=str, default=None, help='path to the data file')
parser.add_argument('--save_dir', type=str, default='./nrt/', help='directory to save the final model')

parser.add_argument('--emsize', type=int, default=300, help='size of embeddings')
parser.add_argument('--nhid', type=int, default=400, help='number of hidden units')
parser.add_argument('--nlayers', type=int, default=4, help='number of layers for rating prediction')
parser.add_argument('--vocab_size', type=int, default=20000, help='keep the most frequent words in the vocabulary')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regularization')
parser.add_argument('--text_reg', type=float, default=1.0, help='regularization on text generation task')
parser.add_argument('--rating_reg', type=float, default=1.0, help='regularization on rating prediction task')
parser.add_argument('--words', type=int, default=128, help='number of words to generate for each sample')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='use CUDA')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--endure_times', type=int, default=100, help='the maximum endure times of loss increasing on validation')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--load_model', action=argparse.BooleanOptionalAction, default=False, help='load model from the checkpoint')

parser.add_argument("--use_explanations", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use explanations for training.")

config = parser.parse_args()

set_seed(config.seed)

model_name = "NRT"
task_type = "review" if not config.use_explanations else "explanation"
config.save_dir = os.path.join(config.save_dir, f"{model_name}_{task_type}", config.dataset_name)

os.makedirs(config.save_dir, exist_ok=True)
config.log_file_path = os.path.join(config.save_dir, "log.txt")
config.save_model_path = os.path.join(config.save_dir, "model.pth")

logger = logging.getLogger("NRT" + config.dataset_name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(config.log_file_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
config.logger = logger

if config.dataset_dir is None:
    parser.error('--data_path should be provided for loading data')

config.logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(config):
    config.logger.info('{:40} {}'.format(arg, getattr(config, arg)))
config.logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not config.cuda:
        config.logger.info('WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if config.cuda else 'cpu')
config.device = device


###############################################################################
# Load data
###############################################################################

train_df, eval_df, test_df, user_id2index, item_id2index = load_data(config)

user_dict = EntityDictionary()
user_dict.entity2idx = user_id2index
user_dict.idx2entity = {v: k for k, v in user_id2index.items()}

item_dict = EntityDictionary()
item_dict.entity2idx = item_id2index
item_dict.idx2entity = {v: k for k, v in item_id2index.items()}

word_dict = build_word_dictionnary(train_df)
word_dict.keep_most_frequent(config.vocab_size)

#corpus = DataLoader(config.data_path, config.index_dir, config.vocab_size)
corpus = DataLoaderFromDataFrameForReviewGeneration(
    user_dict=user_dict, item_dict=item_dict,
    train_df=train_df, eval_df=eval_df, test_df=test_df,
    vocab_size=config.vocab_size, seed=config.seed
)
corpus.max_rating = 5.0
corpus.min_rating = 1.0
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, word2idx, config.words, config.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, config.words, config.batch_size)
test_data = Batchify(corpus.test, word2idx, config.words, config.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(corpus.word_dict)
pad_idx = word2idx['<pad>']
model = NRT(nuser, nitem, ntoken, config.emsize, config.nhid, config.nlayers, corpus.max_rating, corpus.min_rating).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
#optimizer = torch.optim.Adadelta(model.parameters())  # lr is optional to Adadelta

###############################################################################
# Training code
###############################################################################

def train(data):
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.to(device)  # (batch_size, seq_len + 2)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
        r_loss = rating_criterion(rating_p, rating)
        t_loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))
        l2_loss = torch.cat([x.view(-1) for x in model.parameters()]).pow(2.).sum()
        loss = config.text_reg * t_loss + config.rating_reg * r_loss + config.l2_reg * l2_loss
        loss.backward()
        optimizer.step()

        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return text_loss / total_sample, rating_loss / total_sample


def evaluate(data):
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (batch_size, seq_len + 2)
            rating_p, log_word_prob = model(user, item, seq[:, :-1])  # (batch_size,) vs. (batch_size, seq_len + 1, ntoken)
            r_loss = rating_criterion(rating_p, rating)
            t_loss = text_criterion(log_word_prob.view(-1, ntoken), seq[:, 1:].reshape((-1,)))

            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data):
    model.eval()
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            inputs = seq[:, :1].to(device)  # (batch_size, 1)
            hidden = None
            ids = inputs
            for idx in range(config.words):
                # produce a word at each step
                if idx == 0:
                    rating_p, hidden = model.encoder(user, item)
                    rating_predict.extend(rating_p.tolist())
                    log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                else:
                    log_word_prob, hidden = model.decoder(inputs, hidden)  # (batch_size, 1, ntoken)
                #if log_word_prob.dim() == 1:
                #    log_word_prob = log_word_prob.unsqueeze(0)
                word_prob = log_word_prob.squeeze(1).exp()  # (batch_size, ntoken)
                inputs = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
            ids = ids[:, 1:].tolist()  # remove bos
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

    rating_reference = data.rating.tolist()
    rating_scores = rating_evaluation(
        config, predictions=rating_predict, references=rating_reference, 
        metrics=RATING_METRICS,
    )
    with open(os.path.join(config.save_dir, 'rating_results.json'), 'w', encoding='utf-8') as f:
        json.dump(rating_scores, f)
    config.logger.info(str(rating_scores))

    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in test_data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]

    f = lambda t: postprocess_text(t, ['<bos>', '<eos>', '<pad>', '<unk>'])
    text_test = [f(' '.join(tokens)) for tokens in tokens_test]
    text_predict = [f(' '.join(tokens)) for tokens in tokens_predict]
    review_scores = text_evaluation(
        config, predictions=text_predict, references=text_test,
        metrics=TEXT_METRICS
    )
    with open(os.path.join(config.save_dir, 'review_results.json'), 'w', encoding='utf-8') as f:
        json.dump(review_scores, f)
    config.logger.info(str(review_scores))
    return text_test, text_predict

# Loop over epochs.
def trainer():
    global model, config
    best_val_loss = float('inf')
    endure_count = 0
    for epoch in range(1, config.epochs + 1):
        config.logger.info('epoch {}'.format(epoch))
        train_t_loss, train_r_loss = train(train_data)
        config.logger.info('text ppl {:4.4f} | rating loss {:4.4f} | total loss {:4.4f} on train'.format(
            math.exp(train_t_loss), train_r_loss, train_t_loss + train_r_loss))
        val_t_loss, val_r_loss = evaluate(val_data)
        val_loss = val_t_loss + val_r_loss
        config.logger.info('text ppl {:4.4f} | rating loss {:4.4f} | total loss {:4.4f} on validation'.format(
            math.exp(val_t_loss), val_r_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.save_model_path)
        else:
            endure_count += 1
            config.logger.info('Endured {} time(s)'.format(endure_count))
            if endure_count == config.endure_times:
                config.logger.info('Cannot endure it anymore | Exiting from early stop')
                break

trainer()

# Load the best saved model.
model.load(config.save_model_path)
model = model.to(device)

config.logger.info('Generating text')
text_test, text_predict = generate(test_data)
output_df = pd.DataFrame({
    "reference": text_test,
    "prediction": text_predict,
})
output_df.to_csv(os.path.join(config.save_dir, "output.csv"), index=False)
