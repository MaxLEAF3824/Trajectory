import torch
import torch.utils.data as tud
from torch import nn
import numpy as np
import json
import math
import utils

timer = utils.Timer()


class MetricLearningDataset(tud.Dataset):
    def __init__(self, file_train, triplet_num=10):
        """
        train_dict['trajs'] : list of list of idx
        train_dict['sorted_index'] : sorted matrix of most similarity trajs
        train_dict['origin_trajs'] : list of list of (lon, lat)
        """
        self.train_dict = json.load(open(file_train))
        self.triplet_num = triplet_num
        lens = np.array([len(i) for i in self.train_dict['trajs']])
        mask = np.arange(lens.max()) < lens[:, None]
        self.data = np.zeros(mask.shape, dtype=np.int32)
        self.data[mask] = np.concatenate(self.train_dict['trajs'])
        # self.data = np.array([np.array(t) for t in self.train_dict['trajs']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        positive = self.data[self.train_dict['sorted_index'][idx][:self.triplet_num]]
        negative = self.data[self.train_dict['sorted_index'][idx][-self.triplet_num:]]
        return anchor, positive, negative


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, max_len, d_model]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0, 1)
        return self.dropout(x)


class T3S(nn.Module):
    def __init__(self, vocab_size, dim_emb=256, heads=8, encoder_layers=1, lstm_layers=1, pre_emb=None, max_len=128):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        self.max_len = max_len
        if pre_emb is not None:
            self.embedding = nn.Embedding(vocab_size, dim_emb).from_pretrained(pre_emb)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.position_encoding = PositionalEncoding(dim_emb, dropout=0.1, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_emb, heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, encoder_layers)
        self.lstm = nn.LSTM(dim_emb, dim_emb, lstm_layers, batch_first=True)

    def pad(self, x):
        """
        x: [batch_size, seq_len]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            x = torch.cat([x, torch.zeros(batch_size, pad_len).long().to(self.device)], dim=1)
        elif seq_len > self.max_len:
            index, _ = torch.randperm(seq_len)[:self.max_len].sort()
            x = x[:, index]
        return x

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        emb_x = self.embedding(x)  # emb_x: [batch_size, seq_len, dim_emb]
        emb_x = self.pad(emb_x)  # emb_x: [batch_size, max_len, dim_emb]
        emb_x = self.position_encoding(emb_x)  # emb_x: [batch_size, max_len, dim_emb]
        encoder_out = self.encoder(emb_x)  # encoder_out: [batch_size, max_len, dim_emb]
        encoder_out = torch.mean(encoder_out, 1)  # encoder_out: [batch_size, dim_emb]
        lstm_out, (h_n, c_n) = self.lstm(emb_x)  # lstm_out: [batch_size, max_len, dim_emb]
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
        tri_num = positive.shape[1]
        positive = positive.reshape(-1, positive.shape[2])
        negative = negative.reshape(-1, negative.shape[2])
        output_a = self.forward(anchor).unsqueeze(1).repeat(1, tri_num, 1)  # [bsz, triplet_num, emb]
        output_p = self.forward(positive).reshape(batch_size, tri_num, -1)  # [bsz, triplet_num, emb]
        output_n = self.forward(negative).reshape(batch_size, tri_num, -1)  # [bsz, triplet_num, emb]
        dis_pos = torch.norm(output_a - output_p, p=2, dim=2)  # [bsz, triplet_num])
        dis_neg = torch.norm(output_a - output_n, p=2, dim=2)  # [bsz, triplet_num])
        weight = torch.softmax(torch.ones(tri_num) / torch.arange(1, tri_num + 1).float(), dim=0).to(dis_pos.device)
        dis_pos = torch.einsum("bt, t -> b", dis_pos, weight)  # [bsz]
        dis_neg = dis_neg.mean(dim=1)  # [bsz]
        loss = torch.sigmoid(dis_pos) - torch.sigmoid(dis_neg) + 1
        return torch.mean(loss)

    def pair_similarity(self, trajs):
        out = self.forward(trajs)  # [bsz, emb]
        norm = torch.norm(out.unsqueeze(1) - out, dim=2, p=2)  # [bsz, bsz]
        return -norm

    def evaluate(self, test_loader):
        self.eval()
        dataset = test_loader.dataset
        losses = []
        accs = []
        with torch.no_grad():
            for anchor, positive, negative in test_loader:
                losses.append(self.calculate_loss(anchor, positive, negative).item())
            sim_matrix = self.pair_similarity(dataset.data).cpu().numpy()
            sorted_index = np.argsort(-sim_matrix, axis=1)
            for i in range(len(sorted_index)):
                accs.append(len(np.intersect1d(sorted_index[i][:dataset.triplet_num],
                                               dataset[i][1].cpu().numpy())) / dataset.triplet_num)
        loss = np.mean(losses)
        acc = np.mean(accs)
        self.train()
        return loss, acc


def train_t3s(args):
    timer.tik("prepare data")

    # load args
    train_dataset = args.train_dataset
    validate_dataset = args.validate_dataset
    batch_size = args.batch_size
    pretrained_embedding_file = args.pretrained_embedding
    emb_size = args.embedding_size
    learning_rate = args.learning_rate
    epochs = args.epoch_num
    checkpoint = args.checkpoint
    vocab_size = args.vocab_size
    vp = args.visdom_port

    # prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MetricLearningDataset(train_dataset)
    train_loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MetricLearningDataset(validate_dataset)
    test_loader = tud.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # init model
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb = torch.FloatTensor(np.load(pretrained_embedding_file))
    model = T3S(vocab_size=vocab_size, dim_emb=emb_size, pre_emb=pre_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    epoch_start = 0

    # load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        if checkpoint.get('model'):
            model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('epoch'):
            epoch_start = checkpoint['epoch'] + 1

    # init visdom
    if vp != 0:
        from visdom import Visdom
        env = Visdom(port=args.visdom_port)
        pane1 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='train loss'))
        pane2 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='validate loss'))
        pane3 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='accuracy'))
    timer.tok("prepare data")

    # train
    timer.tik("train")
    batch_count = 0
    train_loss_list = []
    validate_loss_list = [0]
    acc_list = [0]
    for epoch in range(epoch_start, epochs):
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            loss = model.calculate_loss(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timer.tok(f"epoch:{epoch} batch:{batch_idx} train loss:{loss.item():.4f}")
            train_loss_list.append(float(loss))
            batch_count += 1
            if vp != 0:
                env.line(X=list(range(batch_count)), Y=train_loss_list, win=pane1)
                env.line(X=list(range(epoch + 1)), Y=validate_loss_list, win=pane2)
                env.line(X=list(range(epoch + 1)), Y=acc_list, win=pane3)
        if epoch % 1 == 0:
            validate_loss, acc = model.evaluate(test_loader, device)
            validate_loss_list.append(validate_loss)
            acc_list.append(acc)
            timer.tok(f"epoch:{epoch} batch:{batch_idx} validate loss:{validate_loss:.4f} acc:{acc:.4f}")
        if epoch % 10 == 1:
            checkpoint = {'model': model.state_dict(), 'optihmizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(checkpoint,
                       f'model/checkpoint_{epoch}_loss{round(float(loss), 3)}_acc_{round(float(acc), 3)}.pth')
