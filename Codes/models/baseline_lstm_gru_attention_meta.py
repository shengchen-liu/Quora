from common import *
import utils


class NeuralNet(nn.Module):
    def __init__(self, config, embedding_matrix):
        super(NeuralNet, self).__init__()

        hidden_size = 60

        self.embedding = nn.Embedding(config.max_features, config.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        # Randomly zero out entire channels (a channel is a 2D feature map, e.g., the
        # jjj-th channel of the iii-th sample in the batched input is a 2D tensor input[i,j]\text{input}[i, j]input[i,j])
        # of the input tensor).
        self.embedding_dropout = nn.Dropout2d(0.1)
        # self.lstm = nn.GRU(config.embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(input_size=config.embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, config.maxlen)
        self.gru_attention = Attention(hidden_size * 2, config.maxlen)

        self.linear = nn.Linear(480, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x, meta):
        '''

        :param x: word tokens
        :param meta: meta features
        :return:
        '''

        h_embedding = self.embedding(x)  # convert questions to embedding vectors
        # unsqueeze: [batch_size, conf.max_len, 300] ->[1, batch_size, config.max_len, 300]
        # dropout -> [1, batch_size, config.max_len, 300] some channels will be zeroed
        # sequeeze: [1, batch_size, config.max_len, 300] - > [batch_size, conf.max_len, 300]
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding) # (batch_size, seq_len, 2*hidden_size)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1) #(batch_size, 2*hidden_size)
        max_pool, _ = torch.max(h_gru, 1) #(batch_size, 2*hidden_size)

        # meta features: (batch_size, num_features)
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, meta), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
