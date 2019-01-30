from common import *


class QuoraDataset(Dataset):
    def __init__(self,mode, x_meta, x_fold, y_fold=0):
        '''

        :param x_fold: torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
        :param y_fold:
        :param x_meta:
        '''
        self.mode = mode
        self.x_fold = x_fold
        if self.mode in ['train', 'valid']:
            self.y_fold = y_fold
        self.x_meta = x_meta


    def __getitem__(self, index):
        # stuff
        ...
        x_fold_item = self.x_fold[index]
        x_meta = self.x_meta[index]
        if self.mode in ['train', 'valid']:
            y_fold_item = self.y_fold[index]
            return (x_fold_item, y_fold_item, x_meta)
        elif self.mode == 'test':
            return (x_fold_item, x_meta)

    def __len__(self):
        return self.x_fold.shape[0]

if __name__ == '__main__':
    batch_size = 5
    input = np.random.uniform(0, 1, (batch_size, 72)).astype(np.float32) # (batch, config.max_len)
    truth_index = np.random.choice(2, batch_size).astype(np.int)
    truth = np.zeros((batch_size, 2))
    for row, i in enumerate(truth_index):
        truth[row, i] = 1
    x_meta = np.random.uniform(0,10, (batch_size, 10)).astype(np.float32) # (batch, number of meta features)
    # ------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()
    x_meta = torch.from_numpy(x_meta).float().cuda()

    train_dataset = QuoraDataset(x_fold=input, y_fold=truth, x_meta=x_meta)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    # train
    for i, (x_batch, y_batch, meta) in enumerate(train_loader):
        print(i)

