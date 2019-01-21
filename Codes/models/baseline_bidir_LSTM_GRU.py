from common import *
import utils
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644
# input shape: (seq_len, batch, input_size)

class model(nn.Module):
    def __init__(self, config, embedding_matrix):
        super(Baseline_Bidir_LSTM_GRU, self).__init__()

        hidden_size = 60  # The number of features in the hidden state h

        self.embedding = nn.Embedding(config.max_features, config.embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.5)
        self.lstm = nn.LSTM(input_size=config.embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        # self.lstm_attention = Attention(hidden_size * 2, config.maxlen)
        # self.gru_attention = Attention(hidden_size * 2, config.maxlen)

        self.linear = nn.Linear(480, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        # 1. embedding
        h_embedding = self.embedding(x)

        # 2. spatial dropout to prevent over-fitting
        #         https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        #       3. Bidirectional LSTM
        # https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        h_lstm, (hn_lstm, cn_lstm) = self.lstm(h_embedding)
#         print("hn_lstm:", hn_lstm.shape)
        #     4. Bidirectional GRU
        h_gru, l_gru = self.gru(h_lstm)

        #       5. A concatenation of the last state, maximum pool, average pool and two features:
        #        "Unique words rate" and "Rate of all-caps words"
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        # conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        hn_lstm = hn_lstm.view(hn_lstm.shape[1],-1)
        l_gru = l_gru.view(l_gru.shape[1],-1)
#         print("l_gru:", l_gru.shape) # torch.Size([2, 1536, 60]) ->[1536, 2*60]
#         print("avg_pool:", avg_pool.shape)
#         print("max_pool:", max_pool.shape)
        conc = torch.cat((hn_lstm, l_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
