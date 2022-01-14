import json
import random

import utils
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import torch.nn as nn
from scipy.spatial.distance import cosine as cosine_dis
import sys
from visdom import Visdom


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
        return self.in_embedding.weight.data.cpu().numpy()


def train_cell2vec(file, window_size, embedding_size, batch_size, epoch_num, learning_rate, checkpoint, pretrained):
    # init
    sys.stdout = utils.Logger('log/train_cell2vec.log')
    timer = utils.Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_rate = 100  # negative sampling rate
    save_rate = 0.9

    timer.tik("read json")
    with open(file) as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}
    timer.tok()

    timer.tik("build dataset")
    dataset = CellEmbeddingDataset(cell2idx, window_size, neg_rate)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    timer.tok()

    timer.tik("training")
    env2 = Visdom()
    pane1 = env2.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(title='loss'))
    model = Cell2Vec(len(cell2idx), embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_start = 0
    model.train()
    last_loss = float('inf')
    loss_list = []
    loss = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        if checkpoint.get('model'):
            model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('epoch'):
            epoch_start = checkpoint['epoch'] + 1
    elif pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    print(f'start time : {timer.now()}\nwindow_size : {dataset.window_size}\nbatch_size : {dataloader.batch_size}'
          f'\nembedding_size : {model.embedding_size}\nepoch_num : {epoch_num}\nlearning_rate : {learning_rate}'
          f'\ndevice : {device}')

    for epoch in range(epoch_start, epoch_num):
        for i, (center, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(center.to(device), positive.to(device), negative.to(device)).mean()
            loss.backward()
            optimizer.step()
            loss_list.append(float(loss))
            if i % 50 == 0 and not (i == 0 and epoch == epoch_start):
                timer.tok(f"epoch:{epoch}, iter:{i}/{len(cell2idx) // batch_size} loss:{loss}")
                if np.mean(loss_list) < save_rate * last_loss:
                    last_loss = np.mean(loss_list)
                    loss_list.clear()
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(checkpoint, f'model/checkpoint_{embedding_size}_{round(float(loss), 3)}.pth')
            env2.line(
                X=np.array([epoch * (len(cell2idx) // batch_size + 1) + i]),
                Y=np.array([float(loss)]),
                win=pane1,  # win参数确认使用哪一个pane
                update='append')
            # evaluate_cell2vec(model.input_embedding(), dataset, window_size)
    embedding_weights = model.input_embedding()
    np.save(f'model/cell_embedding_{embedding_size}_{round(float(loss), 3)}', embedding_weights)


def evaluate_cell2vec(embedding_weights, dataset):
    vocab_size = len(dataset)
    for i in range(10):
        idx = random.randint(0, vocab_size)
        truth = dataset[idx][1].numpy()
        predict = []
        for j in range(embedding_weights.shape[0]):
            if j == idx:
                continue
            dis = cosine_dis(embedding_weights[idx], embedding_weights[j])
            predict.append((j, dis))
        predict.sort(key=lambda x: x[1])
        predict = [t[0] for t in predict[:dataset.window_size]]
        print(len(set(predict) & set(truth)))


if __name__ == "__main__":
    with open('data/str_cell2idx_800.json') as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}

    dataset = CellEmbeddingDataset(cell2idx, 20, 100)
    embedding_weights = np.load('model/cell_embedding_128_0.03.npy')
    vocab_size = len(dataset)
    for i in range(10):
        idx = random.randint(0, vocab_size)
        truth = dataset[idx][1].numpy()
        predict = []
        for j in range(embedding_weights.shape[0]):
            if j == idx:
                continue
            dis = cosine_dis(embedding_weights[idx], embedding_weights[j])
            predict.append((j, dis))
        predict.sort(key=lambda x: x[1])
        predict = [t[0] for t in predict[:dataset.window_size]]
        print(len(set(predict) & set(truth)))
