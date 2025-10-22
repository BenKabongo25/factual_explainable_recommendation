# Att2Seq: Learning to Generate Product Reviews from Attributes
# Paper: https://aclanthology.org/E17-1059.pdf
# Source: https://github.com/lileipisces/Att2Seq


import torch
import torch.nn as nn
import torch.nn.functional as func


class MLPEncoder(nn.Module):

    def __init__(self, n_users, n_items, embedding_dim, hidden_size, n_layers):
        super(MLPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.encoder = nn.Linear(embedding_dim * 2, hidden_size * n_layers)
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, U_ids, I_ids):
        # U_ids, I_ids: (batch_size,)
        U = self.user_embeddings(U_ids) # (batch_size, embedding_dim)
        I = self.item_embeddings(I_ids) # (batch_size, embedding_dim)
        ui_concat = torch.cat([U, I], 1) # (batch_size, embedding_dim * 2)
        hidden = self.tanh(self.encoder(ui_concat))  # (batch_size, hidden_size * n_layers)
        state = hidden.reshape((-1, self.n_layers, self.hidden_size)).permute(1, 0, 2).contiguous()  # (n_layers, batch_size, hidden_size)
        return state


class LSTMDecoder(nn.Module):

    def __init__(self, n_tokens, embedding_dim, hidden_size, n_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(n_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_tokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.08
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, T_ids, ht, ct):  # seq: (batch_size, seq_len), ht & ct: (n_layers, batch_size, hidden_size)
        T_embeddings = self.word_embeddings(T_ids)  # (batch_size, seq_len, embedding_dim)
        output, (ht, ct) = self.lstm(T_embeddings, (ht, ct))  # (batch_size, seq_len, hidden_size) vs. (n_layers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, n_tokens)
        return func.log_softmax(decoded, dim=-1), ht, ct


class Att2Seq(nn.Module):

    def __init__(self, n_users, n_items, n_tokens, embedding_dim, hidden_size, dropout, n_layers=2):
        super(Att2Seq, self).__init__()
        self.encoder = MLPEncoder(n_users, n_items, embedding_dim, hidden_size, n_layers)
        self.decoder = LSTMDecoder(n_tokens, hidden_size, hidden_size, n_layers, dropout)

    def forward(self, U_ids, I_ids, T_ids):  # (batch_size,) vs. (batch_size, seq_len)
        h0 = self.encoder(U_ids, I_ids)  # (n_layers, batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        logits, _, _ = self.decoder(T_ids, h0, c0)
        return logits  # (batch_size, seq_len, n_tokens)
    
    def generate(self, U_ids, I_ids, word_dict, review_length):
        bos_token = word_dict.word2idx['<bos>']
        inputs = torch.tensor([bos_token]).unsqueeze(0).to(U_ids.device)  # (1, 1)
        inputs = inputs.repeat(U_ids.size(0), 1) # (batch_size, 1)
        hidden = None
        hidden_c = None
        ids = inputs
        for idx in range(review_length):
            # produce a word at each step
            if idx == 0:
                hidden = self.encoder(U_ids, I_ids)
                hidden_c = torch.zeros_like(hidden).to(U_ids.device)
                logits, hidden, hidden_c = self.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, n_tokens)
            else:
                logits, hidden, hidden_c = self.decoder(inputs, hidden, hidden_c)  # (batch_size, 1, n_tokens)
            if logits.ndim == 3:
                logits = logits.squeeze(1)  # (batch_size, n_tokens)
            word_prob = logits.exp()  # (batch_size, n_tokens)
            inputs = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
            ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
        ids = ids[:, 1:].tolist()  # remove bos

        reviews_hat = []
        for i in range(len(ids)):
            review = word_dict.ids2tokens(ids[i])
            review = [word for word in review if word not in ['<bos>', '<eos>', '<pad>', '<unk>']]
            reviews_hat.append(" ".join(review))
        return reviews_hat
    
    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))
    
