import json
import time

import utils
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn as nn
import sys


class CellEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, cell2idx: dict, window_size, neg_rate):
        self.window_size = window_size
        self.neg_rate = neg_rate
        self.cell2idx = cell2idx
        self.idx2cell = {cell2idx[c]: c for c in cell2idx}
        sorted_cells = []
        self.cells_arrange = torch.arange(len(cell2idx))
        for i in range(len(cell2idx)):
            sorted_cells.append(self.idx2cell[i])
        self.positive = []
        tree = KDTree(sorted_cells)
        distance, index = tree.query(sorted_cells, k=window_size + 1)
        for i in range(index.shape[0]):
            self.positive.append(index[i, 1:])
        self.positive = torch.tensor(np.array(self.positive))

    def __len__(self):
        return len(self.cell2idx)

    def __getitem__(self, idx):
        ones = torch.ones(len(self.cell2idx))
        ones[idx] = 0
        for i in self.positive[idx]:
            ones[i] = 0
        negative = torch.multinomial(ones, self.neg_rate * self.window_size, replacement=True)
        return self.cells_arrange[idx], self.positive[idx], negative


class Cell2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Cell2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_embedding = nn.Embedding(vocab_size, embedding_size)
        self.out_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, center: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        c_vec = self.in_embedding(center).unsqueeze(2)  # [batch, embedding_size, 1]
        p_vec = self.out_embedding(positive)  # [batch, window_size, embedding_size]
        n_vec = self.out_embedding(negative)  # [batch, window_size * neg_rate, embedding_size]

        p_dot = torch.bmm(p_vec, c_vec)
        n_dot = torch.bmm(n_vec, -c_vec)
        log_pos = F.logsigmoid(p_dot).sum(1)
        log_neg = F.logsigmoid(n_dot).sum(1)
        loss = -(log_pos + log_neg)
        return loss

    def input_embedding(self):  # 取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


def train_cell2vec(file="data/str_cell2idx_800.json", window_size=20, batch_size=512, embedding_size=128, epoch_num=100,
                   learning_rate=1e-3):
    sys.stdout = utils.Logger('log/train_cell2vec.log')
    timer = utils.Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_rate = 100  # negative sampling rate
    print(f'start time : {timer.now()}\nwindow_size : {window_size}\nbatch_size : {batch_size}'
          f'\nembedding_size : {embedding_size}\nepoch_num : {epoch_num}\nlearning_rate : {learning_rate}'
          f'\nnegative sampling rate : {neg_rate}\ndevice : {device}')
    timer.tik("read")
    with open(file) as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}
    timer.tok()

    timer.tik("build dataset")
    dataset = CellEmbeddingDataset(cell2idx, window_size, neg_rate)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    timer.tok()

    model = Cell2Vec(len(cell2idx), embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    timer.tik("training")
    for epoch in range(epoch_num):
        for i, (center, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(center.to(device), positive.to(device), negative.to(device)).mean()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                timer.tok(f"epoch:{epoch}, iter:{i}/{len(cell2idx) // batch_size} loss:{loss}")

    embedding_weights = model.input_embedding()
    np.save('result/embedding-{}'.format(embedding_size), embedding_weights)
    torch.save(model.state_dict(), 'result/embedding-{}.th'.format(embedding_size))


if __name__ == '__main__':
    train_cell2vec()
