import json

from torch.utils.data.dataset import T_co

import args
import utils
from traj2cell import Traj2Cell
from joblib import Parallel, delayed
from copy import copy
import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window_size = 5
batch_size = 8
negative_sampling_num = 100


class CellEmbeddingDataset(Data.Dataset):
    def __init__(self, cell2idx: dict, window_size, negative_sampling_num):
        self.window_size = window_size
        self.negative_sampling_num = negative_sampling_num
        self.cell2idx = cell2idx
        self.idx2cell = {cell2idx[c]: c for c in cell2idx}
        sorted_cells = []
        self.cells_arrange = torch.arange(len(cell2idx)).unsqueeze(1)
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
        negative = torch.multinomial(ones, self.negative_sampling_num * self.window_size, replacement=True)
        return self.cells_arrange[idx], self.positive[idx], negative


class Cell2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_embedding = nn.Embedding(vocab_size, embedding_size)
        self.out_embedding = nn.Embedding(vocab_size, embedding_size)


if __name__ == '__main__':
    timer = utils.Timer()
    timer.tik("read")
    with open('data/str_cell2idx_800.json') as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}
    timer.tok()

    timer.tik("build dataset")
    dataset = CellEmbeddingDataset(cell2idx, window_size, negative_sampling_num)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    timer.tok()

    for a, b, c in dataloader:
        print(a.shape, b.shape, c.shape)
        break
