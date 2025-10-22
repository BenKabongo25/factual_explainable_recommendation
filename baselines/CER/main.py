
import argparse
import ast
import json
import logging
import math
import os
import pandas as pd
import torch
import torch.nn as nn

from module import CER
from baselines.utils import (
    Batchify, DataLoaderFromDataFrameForReviewGeneration, EntityDictionary, 
    RATING_METRICS, TEXT_METRICS,
    build_word_dictionnary, ids2tokens, load_data, postprocess_text, rating_evaluation, set_seed, text_evaluation
)


parser = argparse.ArgumentParser(description='CER (Coherent Explainable Recommender)')

parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
parser.add_argument('--dataset_dir', type=str, default=None, help='path to the data file')
parser.add_argument('--save_dir', type=str, default='./nrt/', help='directory to save the final model')

parser.add_argument('--emsize', type=int, default=512, help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--vocab_size', type=int, default=20000, help='keep the most frequent words in the dict')
parser.add_argument('--rating_reg', type=float, default=0.1, help='regularization on recommendation task')
parser.add_argument('--context_reg', type=float, default=1.0, help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0, help='regularization on text generation task')
parser.add_argument('--peter_mask', action=argparse.BooleanOptionalAction, default=True,
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--use_feature', action=argparse.BooleanOptionalAction, default=False,
                    help='False: no feature; True: use the feature')
parser.add_argument('--words', type=int, default=128, help='number of words to generate for each sample')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='use CUDA')
parser.add_argument('--load_model', action=argparse.BooleanOptionalAction, default=False, help='load model from checkpoint')
parser.add_argument('--lr', type=float, default=1.0, help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--log_interval', type=int, default=200, help='report interval')
parser.add_argument('--endure_times', type=int, default=100, help='the maximum endure times of loss increasing on validation')

parser.add_argument("--use_explanations", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use explanations for training.")

config = parser.parse_args()

set_seed(config.seed)

model_name = "CER"
task_type = "review" if not config.use_explanations else "explanation"
config.save_dir = os.path.join(config.save_dir, f"{model_name}_{task_type}", config.dataset_name)

os.makedirs(config.save_dir, exist_ok=True)
config.log_file_path = os.path.join(config.save_dir, "log.txt")
config.save_model_path = os.path.join(config.save_dir, "model.pth")

logger = logging.getLogger("CER" + config.dataset_name)
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

if config.use_feature:
    src_len = 2 + train_data.feature.size(1)  # [u, i, f]
else:
    src_len = 2  # [u, i]
tgt_len = config.words + 1  # added <bos> or <eos>
ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
pad_idx = word2idx['<pad>']
model = CER(config.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, config.emsize, config.nhead, config.nhid, config.nlayers, config.dropout)
if config.load_model:
    model.load(config.save_model_path) 
model = model.to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

###############################################################################
# Training code
###############################################################################


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
        feature = None
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
        if config.use_feature:
            feature = feature.t().to(device)  # (1, batch_size)
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
        r_loss = rating_criterion(rating_p[0], rating)
        r_loss += rating_criterion(rating_p[1], rating_p[0])
        t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
        loss = config.text_reg * t_loss + config.context_reg * c_loss + config.rating_reg * r_loss
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        context_loss += batch_size * c_loss.item()
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step % config.log_interval == 0 or data.step == data.total_step:
            cur_c_loss = context_loss / total_sample
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            config.logger.info('context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            context_loss = 0.
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq = data.next_batch()  # (batch_size, seq_len), data.step += 1
            feature = None
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
            if config.use_feature:
                feature = feature.t().to(device)  # (1, batch_size)
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
            r_loss = rating_criterion(rating_p[0], rating)
            r_loss += rating_criterion(rating_p[1], rating_p[0])
            t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))

            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq = data.next_batch()
            feature = None
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
            if config.use_feature:
                feature = feature.t().to(device)  # (1, batch_size)
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for idx in range(config.words):
                # produce a word at each step
                if idx == 0:
                    log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_p = rating_p[0]  # (batch_size,)
                    if rating_p.dim() == 0:
                        rating_p = rating_p.unsqueeze(0)
                    rating_predict.extend(rating_p.tolist())
                    context = predict(log_context_dis, topk=config.words)  # (batch_size, words)
                    #print("context", context)
                    context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
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
    print(rating_scores)

    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    f = lambda t: postprocess_text(t, ['<bos>', '<eos>', '<pad>', '<unk>'])
    text_test = [f(' '.join(tokens)) for tokens in tokens_test]
    text_predict = [f(' '.join(tokens)) for tokens in tokens_predict]
    tokens_context = [f(' '.join([idx2word[i] for i in ids])) for ids in context_predict]
    review_scores = text_evaluation(
        config, predictions=text_predict, references=text_test,
        metrics=TEXT_METRICS
    )
    with open(os.path.join(config.save_dir, 'review_results.json'), 'w', encoding='utf-8') as f:
        json.dump(review_scores, f)
    print(review_scores)
    return text_test, text_predict, tokens_context


def trainer():
    # Loop over epochs.
    global model, config, train_data, val_data, scheduler
    best_val_loss = float('inf')
    endure_count = 0
    for epoch in range(1, config.epochs + 1):
        config.logger.info('epoch {}'.format(epoch))
        train(train_data)
        val_c_loss, val_t_loss, val_r_loss = evaluate(val_data)
        if config.rating_reg == 0:
            val_loss = val_t_loss
        else:
            val_loss = val_t_loss + val_r_loss
        config.logger.info('context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
            math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.save_model_path)
        else:
            endure_count += 1
            config.logger.info('Endured {} time(s)'.format(endure_count))
            if endure_count == config.endure_times:
                config.logger.info('Cannot endure it anymore | Exiting from early stop')
                break
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            scheduler.step()
            config.logger.info('Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

trainer()

def test():
    # Load the best saved model.
    global model, config, test_data
    model.load(config.save_model_path)
    model = model.to(config.device)

    # Run on test data.
    test_c_loss, test_t_loss, test_r_loss = evaluate(test_data)
    print('=' * 89)
    config.logger.info('context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
        math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

    config.logger.info('Generating text')
    text_test, text_predict, tokens_context = generate(test_data)
    output_df = pd.DataFrame({
        "reference": text_test,
        "prediction": text_predict,
        "context": tokens_context
    })
    output_df.to_csv(os.path.join(config.save_dir, "output.csv"), index=False)

test()
