from common import *
import utils
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644
# input shape: (seq_len, batch, input_size)

class NeuralNet(nn.Module):
    def __init__(self, config, embedding_matrix):
        super(NeuralNet, self).__init__()

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

def run_check_net():

    batch_size = 32
    C,H,W = 3, 32, 32
    num_class = 5

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()


    #---
    criterion = softmax_cross_entropy_criterion
    net = Net(num_class).cuda()
    net.set_mode('train')
    # print(net)
    ## exit(0)

    net.load_pretrain('../../../model/resnet34-333f7ec4.pth')

    logit = net(input)
    loss  = criterion(logit, truth)
    precision, top = metric(logit, truth)

    print('loss    : %0.8f  '%(loss.item()))
    print('correct :(%0.8f ) %0.8f  %0.8f '%(precision.item(), top[0].item(),top[-1].item()))
    print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    print('        loss  | prec      top      ')
    print('[iter ]       |           1  ... k ')
    print('-------------------------------------')
    while i<=500:

        logit   = net(input)
        loss    = criterion(logit, truth)
        precision, top = metric(logit, truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
            ))
        i = i+1





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')