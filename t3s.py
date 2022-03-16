from logging import raiseExceptions
from timeit import repeat
import torch
from torch import embedding, long, negative, nn
import numpy as np
import json
from traj2grid import *
import random
import traj_dist.distance as tdist


class MetricLearningDataset(torch.utils.data.Dataset):
    def __init__(self, file_train, file_dis_info, grid2idx, metric="edr", max_len=128, triplet_num=10):
        self.metric = metric
        self.triplets_num = triplet_num
        self.saved_triplets = {}
        self.json_obj = json.load(open(file_train))
        self.keys = []
        self.data = []
        # turn the trajectory to max_len
        for key in self.json_obj["traj"].keys():
            traj = np.array(self.json_obj["traj"][key])
            if len(traj) > max_len:
                idx = [0] + sorted(random.sample(range(1, len(traj) - 1), max_len - 2)) + [len(traj) - 1]
                traj = traj[idx]
            elif len(traj) < max_len:
                traj = np.pad(traj, (0, max_len - len(traj)), "constant", constant_values=len(grid2idx))
            self.data.append(traj)
            self.keys.append(key)
        # turn traj to tensor
        self.data = torch.tensor(np.array(self.data), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        # 用于应对多epoch的情况
        positive_idx = []
        negative_idx = []
        if self.saved_triplets.get(idx):
            positive_idx, negative_idx = self.saved_triplets[idx]
        else:
            origin_traj = np.array(self.json_obj["origin_traj"][self.keys[idx]])
            dis = []
            if self.metric == "edr":
                for i, k in enumerate(self.keys):
                    if i != idx:
                        origin_traj_1 = np.array(self.json_obj["origin_traj"][k])
                        dis.append((i, tdist.edr(origin_traj, origin_traj_1, type_d="spherical", eps=21.11)))
                sorted_dis_idx = [t[0] for t in sorted(dis, key=lambda x: x[1])]
                positive_idx = sorted_dis_idx[: self.triplets_num]
                negative_idx = sorted_dis_idx[-self.triplets_num :]
                self.saved_triplets[idx] = (positive_idx, negative_idx)
            else:
                raiseExceptions("metric {} is not supported".format(self.metric))
        positive = self.data[positive_idx]
        negative = self.data[negative_idx]
        return anchor, positive, negative


class T3S(nn.Module):
    def __init__(
        self, vocab_size, dim_emb=256, head_num=8, encoder_layer_num=1, lstm_layer_num=1, pretrain_embedding=None
    ):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # beta for RBF
        nn.init.constant_(self.beta, 0.5)
        if pretrain_embedding:
            self.embedding = nn.Embedding(vocab_size, dim_emb).from_pretrained(pretrain_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_emb, head_num)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, encoder_layer_num)
        self.lstm = nn.LSTM(dim_emb, dim_emb, lstm_layer_num, batch_first=True)

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

    def calculate_loss(self, anchor, positive, negative):
        """
        anchor: [batch_size, seq_len]
        positive: [batch_size, triplet_num, seq_len]
        negative: [batch_size, triplet_num, seq_len]
        """
        batch_size = positive.shape[0]
        triplet_num = positive.shape[1]
        positive = positive.reshape(-1, positive.shape[2])
        negative = negative.reshape(-1, negative.shape[2])
        output_a = self.forward(anchor).unsqueeze(1).repeat(1, triplet_num, 1)  # [bsz, triplet_num, emb]
        output_p = self.forward(positive).reshape(batch_size, triplet_num, -1)  # [bsz, triplet_num, emb]
        output_n = self.forward(negative).reshape(batch_size, triplet_num, -1)  # [bsz, triplet_num, emb]
        sim_pos = torch.exp(-torch.norm(output_a - output_p, p=2, dim=2) / (2 * self.beta))  # [bsz, triplet_num])
        sim_neg = torch.exp(-torch.norm(output_a - output_n, p=2, dim=2) / (2 * self.beta))  # [bsz, triplet_num])
        weight_pos = torch.softmax(torch.ones(triplet_num) / torch.arange(1, triplet_num + 1).float(), dim=0).to(
            sim_pos.device
        )  # [triplet_num]
        sim_pos = torch.einsum("bt, t -> b", sim_pos, weight_pos)  # [bsz]
        # sim_pos = sim_pos.sum(dim=1)
        sim_neg = sim_neg.mean(dim=1)  # [bsz]
        loss = 1 - sim_pos + sim_neg
        return torch.mean(loss)


def train_t3s(model, train_loader, optimizer, epochs=10, device="cuda"):
    # init
    model.to(device)
    model.train()
    for epoch in range(epochs):
        # train
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device)  # anchor: [batch_size, seq_len]
            positive = positive.to(device)  # positive: [batch_size, triplet_num, seq_len]
            negative = negative.to(device)  # negative: [batch_size, triplet_num, seq_len]
            loss = model.calculate_loss(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                pass
            print(f"epoch:{epoch} batch:{batch_idx} loss:{loss.item():.4f}")
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(checkpoint, f'model/checkpoint_{epoch}_{i}_{round(float(loss), 3)}.pth')    


if __name__ == "__main__":
    dataset_dir = "/home/yqguo/coding/Trajectory/data/"
    with open(dataset_dir + "str_grid2idx_400.json") as f:
        str_grid2idx = json.load(f)
        f.close()
    grid2idx = {eval(c): str_grid2idx[c] for c in list(str_grid2idx)}
    dataset = MetricLearningDataset(dataset_dir + "1m_gps_20161101_reformat.json", grid2idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    model = T3S(vocab_size=len(grid2idx) + 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_t3s(model, dataloader, optimizer)
