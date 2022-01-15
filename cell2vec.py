import json
import os
import random
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
        return self.in_embedding.weight.data.cpu().numpy()


def train_cell2vec(file, window_size, embedding_size, batch_size, epoch_num, learning_rate, checkpoint, pretrained,
                   visdom_port):
    # init
    timer = utils.Timer()
    sys.stdout = utils.Logger(f'log/train_cell2vec_{timer.now()}.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_rate = 100  # negative sampling rate

    # read dict
    timer.tik("read json")
    with open(file) as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}
    timer.tok()

    # build dataset
    timer.tik("build dataset")
    dataset = CellEmbeddingDataset(cell2idx, window_size, neg_rate)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    timer.tok()

    # training preparation
    model = Cell2Vec(len(cell2idx), embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    cp_save_rate = 0.8
    np_save_rate = 0.95
    iter_num = len(dataset) // batch_size + 1
    epoch_start = 0
    best_loss = float('inf')
    loss_list = []
    best_accuracy = 0
    last_save_epoch = 0

    # start visdom
    if visdom_port != 0:
        from visdom import Visdom
        env = Visdom(port=visdom_port)
        pane1 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='loss'))
        pane2 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='accuracy'))

    # load checkpoint / pretrained_state_dict
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        if checkpoint.get('model'):
            model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('epoch'):
            epoch_start = checkpoint['epoch'] + 1
            last_save_epoch = epoch_start
    elif pretrained is not None:
        model.load_state_dict(torch.load(pretrained))

    # start training
    print(f'\n-------------training config-------------\n'
          f'start time : {timer.now()}\n'
          f'window_size : {dataset.window_size}\n'
          f'batch_size : {dataloader.batch_size}\n'
          f'embedding_size : {model.embedding_size}\n'
          f'epoch_num : {epoch_num}\n'
          f'learning_rate : {learning_rate}\n'
          f'device : {device}\n')
    timer.tik("training")
    for epoch in range(epoch_start, epoch_num):
        for i, (center, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(center.to(device), positive.to(device), negative.to(device)).mean()
            loss.backward()
            optimizer.step()
            loss_list.append(float(loss))
            if visdom_port != 0:
                env.line(
                    X=np.array([(epoch - epoch_start) * iter_num + i]),
                    Y=np.array([float(loss)]),
                    win=pane1,  # win参数确认使用哪一个pane
                    update='append')
            if i % (iter_num // 4 + 1) == 0:
                acc = evaluate_cell2vec(model.input_embedding(), dataset, test_num=100)
                timer.tok(f"epoch:{epoch} iter:{i}/{iter_num} loss:{round(float(loss), 3)} acc:{round(acc, 3)}")
                if i == 0 and epoch == epoch_start:
                    best_loss = np.mean(loss_list)
                    best_accuracy = acc
                    continue
                if np.mean(loss_list) < cp_save_rate * best_loss:
                    best_loss = np.mean(loss_list)
                    loss_list.clear()
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(checkpoint, f'model/checkpoint_{embedding_size}_{epoch}_{i}_{round(float(loss), 3)}.pth')
                elif epoch - last_save_epoch > 5:
                    last_save_epoch = epoch
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(checkpoint, f'model/checkpoint_{embedding_size}_{epoch}_{i}_{round(float(loss), 3)}.pth')
                if visdom_port != 0:
                    env.line(
                        X=np.array([(epoch - epoch_start) * iter_num + i]),
                        Y=np.array([acc]),
                        win=pane2,
                        update='append')
                if acc * np_save_rate > best_accuracy:
                    best_accuracy = acc
                    np.save(f'model/cell_embedding_{embedding_size}_{round(acc, 2)}', model.input_embedding())


def evaluate_cell2vec(embedding_weights, dataset, test_num=10):
    random.seed(20000221)
    samples_index = random.sample(range(len(dataset)), test_num)
    samples_weights = embedding_weights[samples_index, :]

    from scipy.spatial.distance import cdist
    nearest_index = cdist(samples_weights, embedding_weights, metric='cosine').argsort(axis=1)

    predict = list(nearest_index[:, 1:dataset.window_size + 1])
    truth = [dataset[idx][1].numpy() for idx in samples_index]
    accuracy = np.mean([len(np.intersect1d(predict[i], truth[i])) for i in range(test_num)]) / dataset.window_size
    return accuracy

