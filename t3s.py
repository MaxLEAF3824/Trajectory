import torch
import torch.utils.data as tud
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import numpy as np
import json
import math
import utils
import random
from pytorch_metric_learning import losses, distances

timer = utils.Timer()


class MetricLearningDataset(tud.Dataset):
    def __init__(self, file_train, triplet_num, min_len, max_len, dataset_size=None):
        self.triplet_num = triplet_num
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_size = dataset_size
        self.trajs = None
        self.original_trajs = None
        self.dis_matrix = None
        self.sorted_index = None
        self.sim_matrix = None
        self.data_prepare(json.load(open(file_train)))

    def data_prepare(self, train_dict):
        """
        train_dict['trajs'] : list of list of idx
        train_dict['origin_trajs'] : list of list of (lon, lat)
        train_dict['dis_matrix'] : distance matrix
        """
        trajs = []
        original_trajs = []
        used_idxs = []
        x = []
        y = []
        for original_traj in train_dict["origin_trajs"]:
            x.extend([i[0] for i in original_traj])
            y.extend([i[1] for i in original_traj])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        for idx, traj in enumerate(train_dict["trajs"]):
            if self.min_len < len(traj) < self.max_len:
                trajs.append(traj)
                original_traj = train_dict["origin_trajs"][idx]
                original_traj = [[(i[0] - meanx)/stdx, (i[1] - meany)/stdy] for i in original_traj]
                original_trajs.append(original_traj)
                used_idxs.append(idx)
        if self.dataset_size is None:
            self.dataset_size = len(used_idxs)
        else:
            self.dataset_size = min(self.dataset_size, len(used_idxs))
        used_idxs = used_idxs[:self.dataset_size]
        self.trajs = np.array(trajs[:self.dataset_size], dtype=object)
        self.original_trajs = np.array(original_trajs[:self.dataset_size], dtype=object)
        self.dis_matrix = np.array(train_dict["dis_matrix"])[used_idxs, :][:, used_idxs]
        self.sorted_index = np.argsort(self.dis_matrix, axis=1)
        a = 20
        self.sim_matrix = np.exp(-a * self.dis_matrix)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        anchor = self.trajs[idx]
        positive_idx = self.sorted_index[idx][:self.triplet_num+1].tolist()
        if idx in positive_idx:
            positive_idx.remove(idx)
        else:
            positive_idx = positive_idx[:self.triplet_num]
        positive = self.trajs[positive_idx]
        negative_idx = self.sorted_index[idx][-self.triplet_num:].tolist()
        list.reverse(negative_idx)
        negative_idx = np.random.choice(self.sorted_index[idx][self.triplet_num:], self.triplet_num).tolist()
        negative = self.trajs[negative_idx]
        trajs_a = self.original_trajs[idx]
        trajs_p = self.original_trajs[positive_idx]
        trajs_n = self.original_trajs[negative_idx]
        return anchor, positive, negative, trajs_a, trajs_p, trajs_n, idx, positive_idx, negative_idx, self.sim_matrix[idx, positive_idx], self.sim_matrix[idx, negative_idx], self.sim_matrix[idx, :]


def collate_fn(train_data):
    batch_size = len(train_data)
    anchor = [torch.tensor(traj[0]) for traj in train_data]
    anchor_lens = [len(traj) for traj in anchor]
    anchor = rnn_utils.pad_sequence(anchor, batch_first=True, padding_value=-1)

    pos = []
    for list_pos in [list(traj[1]) for traj in train_data]:
        pos.extend(list_pos)
    pos = [torch.tensor(pos_) for pos_ in pos]
    pos_lens = [len(pos_) for pos_ in pos]
    pos = rnn_utils.pad_sequence(pos, batch_first=True, padding_value=-1)

    neg = []
    for list_neg in [list(traj[2]) for traj in train_data]:
        neg.extend(list_neg)
    neg = [torch.tensor(neg_) for neg_ in neg]
    neg_lens = [len(neg_) for neg_ in neg]
    neg = rnn_utils.pad_sequence(neg, batch_first=True, padding_value=-1)

    trajs_a = [torch.tensor(np.array(traj[3]), dtype=torch.float32) for traj in train_data]
    trajs_a_lens = [traj.shape[0] for traj in trajs_a]
    trajs_a = rnn_utils.pad_sequence(trajs_a, batch_first=True, padding_value=0)

    trajs_p = []
    for list_trajs_p in [list(traj[4]) for traj in train_data]:
        trajs_p.extend(list_trajs_p)
    trajs_p = [torch.tensor(traj_p, dtype=torch.float32) for traj_p in trajs_p]
    trajs_p_lens = [traj.shape[0] for traj in trajs_p]
    trajs_p = rnn_utils.pad_sequence(trajs_p, batch_first=True, padding_value=0)

    trajs_n = []
    for list_trajs_n in [list(traj[5]) for traj in train_data]:
        trajs_n.extend(list_trajs_n)
    trajs_n = [torch.tensor(traj_n, dtype=torch.float32) for traj_n in trajs_n]
    trajs_n_lens = [traj.shape[0] for traj in trajs_n]
    trajs_n = rnn_utils.pad_sequence(trajs_n, batch_first=True, padding_value=0)

    anchor_idxs = torch.tensor([traj[6] for traj in train_data], dtype=torch.long)

    pos_idxs = []
    for list_pos_idx in [list(traj[7]) for traj in train_data]:
        pos_idxs.extend(list_pos_idx)
    pos_idxs = torch.tensor([pos_ for pos_ in pos_idxs])

    neg_idxs = []
    for list_neg_idx in [list(traj[8]) for traj in train_data]:
        neg_idxs.extend(list_neg_idx)
    neg_idxs = torch.tensor([neg_ for neg_ in neg_idxs])

    sim_pos = torch.tensor(np.array([traj[9] for traj in train_data]), dtype=torch.float32)
    sim_neg = torch.tensor(np.array([traj[10] for traj in train_data]), dtype=torch.float32)
    sim_matrix_a = torch.tensor(np.array([traj[11] for traj in train_data]), dtype=torch.float32)
    sim_matrix_a = sim_matrix_a[:, anchor_idxs]
    return anchor, anchor_lens, pos, pos_lens, neg, neg_lens, trajs_a,\
        trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,\
        anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(512, emb_size)  # [512, d_model]
        position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)  # [512, 1]
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
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
    def __init__(self, vocab_size, emb_size, heads=8, encoder_layers=1, lstm_layers=1, pre_emb=None):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        if pre_emb is not None:
            self.embedding = nn.Embedding(vocab_size, emb_size).from_pretrained(pre_emb)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size, dropout=0.1, )
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            emb_size, heads, batch_first=True), encoder_layers)
        self.lstm = nn.LSTM(2, emb_size, lstm_layers, batch_first=True)

    def forward(self, x, trajs, trajs_lens):
        """
        x: [batch_size, seq_len]
        trajs: [batch_size, seq_len, 2]
        trajs_lens: [batch_size]
        """
        # transformer embedding
        mask_x = (x == -1)  # [batch_size, seq_len]
        x = x.clamp_min(0)
        emb_x = self.embedding(x)  # emb_x: [batch_size, seq_len, dim_emb]
        pe_emb_x = self.position_encoding(emb_x)
        # encoder_out2 = self.encoder(pe_emb_x, src_key_padding_mask=mask_x)  # [batch_size, seq_len, dim_emb]
        # pe_emb_x[mask_x] = 0
        encoder_out = self.encoder(pe_emb_x, src_key_padding_mask=mask_x)  # [batch_size, seq_len, dim_emb]
        encoder_out_mean = torch.mean(encoder_out, 1)  # batch_size, dim_emb]
        # lstm embedding
        trajs = rnn_utils.pack_padded_sequence(trajs, trajs_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(trajs)
        # lstm_out, out_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        # out_len = out_len.to(lstm_out.device)
        # lstm_out = lstm_out.index_select(1, out_len - 1)
        # lstm_out_last = lstm_out.diagonal(dim1=0, dim2=1).transpose(0, 1)
        # add
        output = self.lamb * encoder_out_mean + (1 - self.lamb) * h_n.squeeze()  # output: [batch_size, dim_emb]
        return output

    def calculate_loss(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                       trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                       anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a):
        batch_size = anchor.size(0)
        tri_num = neg.shape[0] // batch_size
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        # output_n = self.forward(neg, trajs_n, trajs_n_lens)
        sim_p = torch.exp(-torch.norm(output_a.repeat(tri_num, 1) - output_p, dim=1)).reshape(batch_size, -1)
        sim_a = torch.exp(-torch.norm(output_a.unsqueeze(1)-output_a, dim=2)).reshape(batch_size, -1)
        # sim_n = torch.exp(-torch.norm(output_a.repeat(tri_num, 1) - output_n, dim=1)).reshape(batch_size, -1)
        w_p = torch.softmax(torch.ones(tri_num)/torch.arange(1, tri_num+1).float(), dim=0).to(output_a.device)
        loss_p = torch.sum(w_p * (sim_p - sim_pos)**2, dim=1)
        loss_n = torch.sum((torch.relu(sim_a - sim_matrix_a))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss

    def evaluate(self, test_loader, device, tri_num):
        self.eval()
        accs = []
        with torch.no_grad():
            for (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                 trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                 anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a) in test_loader:
                test_num = 200
                tb_anchor = anchor[:test_num].to(device)
                tb_trajs_a = trajs_a[:test_num].to(device)
                tb_trajs_a_lens = trajs_a_lens[:test_num]
                tb_pos_idxs = pos_idxs[:test_num*tri_num].reshape(test_num, tri_num).cpu().numpy()
                output_a = self.forward(tb_anchor, tb_trajs_a, tb_trajs_a_lens)
                bsz = 200
                sim_matrixs = []
                for i in range(len(anchor)//bsz+1):
                    lb = i * bsz
                    ub = min((i+1)*bsz, len(anchor))
                    output_b = self.forward(anchor[lb:ub].to(device), trajs_a[lb:ub].to(device), trajs_a_lens[lb:ub])
                    s = torch.exp(-torch.norm(output_a.unsqueeze(1) - output_b, dim=-1))
                    sim_matrixs.append(s)
                sim_matrix = torch.cat(sim_matrixs, dim=1).cpu().numpy()
                sorted_index = np.argsort(-sim_matrix, axis=1)
                for i in range(test_num):
                    avg_rank = 0
                    for idx in tb_pos_idxs[i]:
                        avg_rank += np.argwhere(sorted_index[i] == idx)[0][0]
                    avg_rank /= len(tb_pos_idxs[i])
                    accs.append(avg_rank)
                break
        acc = np.mean(accs)
        self.train()
        return acc


def train_t3s(args):

    # load args
    train_dataset = args.train_dataset
    validate_dataset = args.validate_dataset
    batch_size = args.batch_size
    pretrained_embedding_file = args.pretrained_embedding
    emb_size = args.embedding_size
    learning_rate = args.learning_rate
    epochs = args.epoch_num
    cp = args.checkpoint
    vocab_size = args.vocab_size
    triplet_num = args.triplet_num
    min_len = args.min_len
    max_len = args.max_len
    heads = args.heads
    device = torch.device(args.device)
    vp = args.visdom_port

    # prepare data
    timer.tik("prepare data")
    dataset = MetricLearningDataset(train_dataset, triplet_num, min_len, max_len)
    train_loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataset = MetricLearningDataset(validate_dataset, triplet_num, min_len, max_len)
    test_loader = tud.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn)
    timer.tok("prepare data")

    # init model
    timer.tik("init model")
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb = torch.FloatTensor(np.load(pretrained_embedding_file))
    model = T3S(vocab_size, emb_size, heads, pre_emb=pre_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    epoch_start = 0
    timer.tok("init model")

    # load checkpoint
    if cp is not None:
        cp = torch.load(cp)
        if cp.get('model'):
            model.load_state_dict(cp['model'])
        if cp.get('optimizer'):
            optimizer.load_state_dict(cp['optimizer'])
        if cp.get('epoch'):
            epoch_start = cp['epoch'] + 1

    # init visdom
    if vp != 0:
        from visdom import Visdom
        env = Visdom(port=args.visdom_port)
        pane1 = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='train loss'))
        pane3 = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='accuracy'))

    # train
    timer.tik("train")
    batch_count = 0
    for epoch in range(epoch_start, epochs):
        if epoch % 1 == 0:
            acc = model.evaluate(test_loader, device, tri_num=triplet_num)
            if vp != 0:
                env.line(X=[epoch + 1], Y=[acc], win=pane3, update='append')
            timer.tok(f"epoch:{epoch} acc:{acc:.4f}")
        for batch_idx, (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                        trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                        anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a) in enumerate(train_loader):
            anchor = anchor.to(device)
            pos = pos.to(device)
            # neg = neg.to(device)
            # trajs_n = trajs_n.to(device)
            trajs_a = trajs_a.to(device)
            trajs_p = trajs_p.to(device)
            sim_pos = sim_pos.to(device)
            # sim_neg = sim_neg.to(device)
            sim_matrix_a = sim_matrix_a.to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(
                anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                anchor_idxs, pos_idxs, neg_idxs, sim_pos, sim_neg, sim_matrix_a)
            loss.backward()
            optimizer.step()
            timer.tok(f"epoch:{epoch} batch:{batch_idx} train loss:{loss.item()}")
            batch_count += 1
            if vp != 0:
                env.line(X=[batch_count], Y=[loss.item()], win=pane1, update='append')
        if epoch % 10 == 9:
            cp = {'model': model.state_dict(), 'optihmizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(cp, f'model/cp_{epoch}_loss{round(float(loss), 3)}_acc_{round(float(acc), 3)}.pth')
