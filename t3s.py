import torch
from torch import nn
import numpy as np
import json
from traj2grid import *
import random
import sys


class MetricLearningDataset(torch.utils.data.Dataset):
    def __init__(self, file_train, grid2idx, metric="lcss", max_len=512):
        json_obj = json.load(open(file_train))
        self.grid2idx = grid2idx
        self.origin_data = list(json_obj["origin_traj"].values())
        self.data = list(json_obj["traj"].values())
        # turn the trajectory to max_len
        for i in range(len(self.data)):
            traj = np.array(self.data[i])
            if len(traj) > max_len:
                idx = [0] + sorted(random.sample(range(1, len(traj) - 1), max_len - 2)) + [len(traj) - 1]
                traj = traj[idx]
            elif len(traj) < max_len:
                traj = np.pad(traj, (0, max_len - len(traj)), "constant", constant_values=0)
            self.data[i] = traj
        # turn traj to tensor
        self.data = torch.tensor(np.array(self.data), dtype=torch.float)
        self.metric = metric

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class T3S(nn.Module):
    def __init__(self, vocab_size, dim_emb=256, head_num=8, layer_num=6, num_layers=1, pretrain_embedding=None):
        super(T3S, self).__init__()
        self.lamb = 0.5  # lambda
        if pretrain_embedding:
            self.embedding = nn.Embedding(vocab_size, dim_emb).from_pretrained(pretrain_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_emb, head_num)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_num)
        self.lstm = nn.LSTM(dim_emb, dim_emb, num_layers, batch_first=True)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        emb_x = self.embedding(x)  # emb_x: [batch_size, seq_len, dim_emb]
        encoder_out = self.encoder(emb_x)  # encoder_out: [batch_size, seq_len, dim_emb]
        encoder_out = torch.mean(encoder_out, 1)  # encoder_out: [batch_size, dim_emb]
        lstm_out, (h_n, c_n) = self.lstm(emb_x)  # lstm_out: [batch_size, seq_len, dim_emb]
        lstm_out = lstm_out[:, -1, :]  # lstm_out: [batch_size, dim_emb]
        output = self.lamb * encoder_out + (1 - self.lamb) * lstm_out  # output: [batch_size, dim_emb]
        return output


def Loss_NCE(output: torch.Tensor, x: torch.Tensor):
    return 0


def Trainer(model, train_loader, optimizer, epochs=10, device="cuda"):
    # init
    timer = utils.Timer()
    sys.stdout = utils.Logger(f"log/train_grid2vec_{timer.now()}.log")

    model.to(device)
    model.train()
    for epoch in range(epochs):
        # train
        for idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = Loss_NCE(output, x)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, idx: {idx}, loss: {loss}")


if __name__ == "__main__":
    dataset_dir = "/home/yqguo/coding/Trajectory/data/"
    with open(dataset_dir + "str_grid2idx_800.json") as f:
        str_grid2idx = json.load(f)
        f.close()
    grid2idx = {eval(c): str_grid2idx[c] for c in list(str_grid2idx)}
    dataset = MetricLearningDataset(dataset_dir + "100k_gps_20161101_reformat.json", grid2idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    T3S = T3S(vocab_size=100).cuda()
