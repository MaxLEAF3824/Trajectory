from lib2to3.pytree import NegatedPattern
from turtle import distance
import torch
import torch.utils.data as tud
import torch.nn.utils.rnn as rnn_utils
from torch import negative_, nn
import numpy as np
import json
import math
import utils
import random
from pytorch_metric_learning import losses, distances

timer = utils.Timer()


class MetricLearningDataset(tud.Dataset):
    def __init__(self, file_train, triplet_num, max_len, negative_rate=0.2):
        """
        train_dict['trajs'] : list of list of idx
        train_dict['sorted_index'] : sorted matrix of most similarity trajs
        train_dict['origin_trajs'] : list of list of (lon, lat)
        train_dict['dis_matrix'] : distance matrix
        """
        self.train_dict = json.load(open(file_train))
        self.triplet_num = triplet_num
        self.negative_num = int(negative_rate*len(self.train_dict['trajs']))
        self.dis_matrix = np.array(self.train_dict['dis_matrix'])
        self.trajs = np.array(self.train_dict["trajs"], dtype=object)
        self.original_traj = np.array(self.train_dict["origin_trajs"], dtype=object)

    def __len__(self):
        return len(self.train_dict["trajs"])

    def __getitem__(self, idx):
        anchor = self.trajs[idx]
        positive_idx = self.train_dict['sorted_index'][idx][:self.triplet_num+1]
        if idx in positive_idx:
            positive_idx.remove(idx)
        else:
            positive_idx = positive_idx[:self.triplet_num]
        positive_idx = random.sample(positive_idx, self.triplet_num)
        positive = self.trajs[positive_idx]
        negative_idx = self.train_dict['sorted_index'][idx][-self.negative_num:]
        negative_idx = random.sample(negative_idx, self.triplet_num)
        negative = self.trajs[negative_idx]
        trajs_a = self.original_traj[idx]
        trajs_p = self.original_traj[positive_idx]
        trajs_n = self.original_traj[negative_idx]
        return anchor, positive, negative, trajs_a, trajs_p, trajs_n, idx, positive_idx, negative_idx, self.dis_matrix[idx, positive_idx], self.dis_matrix[idx, negative_idx]


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

    dis_pos = torch.tensor(np.array([traj[9] for traj in train_data]), dtype=torch.float32)
    dis_neg = torch.tensor(np.array([traj[10] for traj in train_data]), dtype=torch.float32)
    return anchor, anchor_lens, pos, pos_lens, neg, neg_lens, trajs_a,\
        trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,\
        anchor_idxs, pos_idxs, neg_idxs, dis_pos, dis_neg


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_size)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
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
    def __init__(self, vocab_size, emb_size, max_len, heads=8, encoder_layers=1, lstm_layers=1, pre_emb=None):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        if pre_emb is not None:
            self.embedding = nn.Embedding(vocab_size, emb_size).from_pretrained(pre_emb)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size, max_len, dropout=0.1, )
        self.encoder_layer = nn.TransformerEncoderLayer(emb_size, heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, encoder_layers)
        self.lstm = nn.LSTM(2, emb_size, lstm_layers, batch_first=True)
        # self.loss_func = nn.CosineEmbeddingLoss(margin=0.5,reduction='mean')
        # self.dis_func = distances.CosineSimilarity()
        # self.loss_func = losses.NPairsLoss(distance=self.dis_func)

    def forward(self, x, trajs, trajs_lens):
        """
        x: [batch_size, seq_len]
        trajs: [batch_size, seq_len, 2]
        trajs_lens: [batch_size]
        """
        # transformer embedding
        mask_x = (x == -1)  # [batch_size, seq_len]
        x.clamp_min_(0)
        emb_x = self.embedding(x)  # emb_x: [batch_size, seq_len, dim_emb]
        emb_x = self.position_encoding(emb_x)
        # emb_x[mask_x] = 0
        encoder_out = self.encoder(emb_x, src_key_padding_mask=mask_x.transpose(0, 1))  # [batch_size, seq_len, dim_emb]
        encoder_out = torch.mean(encoder_out, 1)  # batch_size, dim_emb]
        # lstm embedding
        trajs = rnn_utils.pack_padded_sequence(trajs, trajs_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(trajs)
        lstm_out, out_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        out_len = out_len.to(lstm_out.device)
        lstm_out = lstm_out.index_select(1, out_len - 1)
        lstm_out = lstm_out.diagonal(dim1=0, dim2=1).transpose(0, 1)
        # 相加
        output = encoder_out
        output = self.lamb * encoder_out + (1 - self.lamb) * lstm_out  # output: [batch_size, dim_emb]
        return output

    def calculate_loss(self, anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                       trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                       anchor_idxs, pos_idxs, neg_idxs, dis_pos, dis_neg):
        """
        anchor: [batch_size, seq_len]
        pos: [batch_size, seq_len]
        neg: [batch_size * triplet_num, seq_len]
        """
        batch_size = anchor.size(0)
        loss = 0
        tri_num = neg.shape[0] // batch_size
        output_a = self.forward(anchor, trajs_a, trajs_a_lens)
        output_p = self.forward(pos, trajs_p, trajs_p_lens)
        output_n = self.forward(neg, trajs_n, trajs_n_lens)
        # loss = self.loss_func(embeddings, labels)
        # dis_p = torch.norm(output_a - output_p, p=2, dim=1)
        # dis_n = torch.norm(output_a.repeat(tri_num, 1) - output_n, p=2, dim=1)
        # loss = dis_p.mean() - dis_n.mean()
        # embeddings = torch.cat([output_p, output_n], dim=0)  # [batch_size * (tri_num+2), dim_emb]
        # targets = -torch.ones((tri_num*2) * batch_size, device=output_a.device)
        # targets[:batch_size*tri_num] = 1
        # loss = self.loss_func(output_a.repeat((tri_num *2), 1), embeddings, targets)
        dis_p = (1 - torch.cosine_similarity(output_a.repeat(tri_num, 1), output_p, dim=1)).reshape(batch_size, -1)
        dis_n = (1 - torch.cosine_similarity(output_a.repeat(tri_num, 1), output_n, dim=1)).reshape(batch_size, -1)
        w_p = (torch.ones(tri_num)/torch.arange(1, tri_num+1).float()).to(output_a.device)
        w_p = torch.softmax(w_p, dim=0)
        loss_p = torch.sum(w_p * (dis_p - dis_pos)**2, dim=1)
        loss_n = torch.sum(w_p * (torch.relu(dis_n - dis_neg))**2, dim=1)
        loss = (loss_p + loss_n).mean()
        return loss

    def evaluate(self, test_loader, device, tri_num, eval_rate=0.1):
        self.eval()
        dataset = test_loader.dataset
        losses = []
        accs = []
        compare_num = int(eval_rate * len(dataset))
        with torch.no_grad():
            for (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                 trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                 anchor_idxs, pos_idxs, neg_idxs, dis_pos, dis_neg) in test_loader:
                anchor = anchor.to(device)
                # pos = pos.to(device)
                # neg = neg.to(device)
                trajs_a = trajs_a.to(device)
                # trajs_p = trajs_p.to(device)
                # trajs_n = trajs_n.to(device)
                output_a = self.forward(anchor, trajs_a, trajs_a_lens)
                dis_func = distances.CosineSimilarity()
                sim_matrix = dis_func(output_a)
                sorted_index = torch.argsort(-sim_matrix, dim=1).cpu().numpy()
                for i in range(len(sorted_index)):
                    acc = np.intersect1d(
                        sorted_index[i][1: tri_num + 1],
                        dataset.train_dict['sorted_index'][i][1: tri_num + 1])
                    accs.append(len(acc) / tri_num)
                break
        loss = 0
        acc = np.mean(accs)
        # acc = 0
        self.train()
        return loss, acc


def train_t3s(args):

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
    triplet_num = args.triplet_num
    max_len = args.max_len
    heads = args.heads
    device = torch.device(args.device)
    vp = args.visdom_port

    # prepare data
    timer.tik("prepare data")
    dataset = MetricLearningDataset(train_dataset, triplet_num, max_len)
    train_loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataset = MetricLearningDataset(validate_dataset, triplet_num, max_len)
    test_loader = tud.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_fn)
    timer.tok("prepare data")

    # init model
    timer.tik("init model")
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb = torch.FloatTensor(np.load(pretrained_embedding_file))
    model = T3S(vocab_size, emb_size, max_len, heads, pre_emb=pre_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    epoch_start = 0
    timer.tok("init model")

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
        pane1 = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='train loss'))
        # pane2 = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='validate loss'))
        pane3 = env.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='accuracy'))

    # train
    timer.tik("train")
    batch_count = 0
    train_loss_list = []
    for epoch in range(epoch_start, epochs):
        for batch_idx, (anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                        trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                        anchor_idxs, pos_idxs, neg_idxs, dis_pos, dis_neg) in enumerate(train_loader):
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            trajs_n = trajs_n.to(device)
            trajs_a = trajs_a.to(device)
            trajs_p = trajs_p.to(device)
            dis_pos = dis_pos.to(device)
            dis_neg = dis_neg.to(device)
            loss = model.calculate_loss(
                anchor, anchor_lens, pos, pos_lens, neg, neg_lens,
                trajs_a, trajs_a_lens, trajs_p, trajs_p_lens, trajs_n, trajs_n_lens,
                anchor_idxs, pos_idxs, neg_idxs, dis_pos, dis_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timer.tok(
                f"epoch:{epoch} batch:{batch_idx} train loss:{loss.item():.4f}")
            train_loss_list.append(float(loss))
            batch_count += 1
            if vp != 0:
                env.line(X=[batch_count], Y=[loss.item()], win=pane1, update='append')
            # validate_loss, acc = model.evaluate(test_loader, device, tri_num=triplet_num)
        if epoch % 1 == 0:
            validate_loss, acc = model.evaluate(test_loader, device, tri_num=triplet_num)
            # env.line(X=[epoch + 1], Y=[validate_loss], win=pane2, update='append')
            env.line(X=[epoch + 1], Y=[acc], win=pane3, update='append')
            timer.tok(f"epoch:{epoch} batch:{batch_idx} validate loss:{validate_loss:.4f} acc:{acc:.4f}")
        if epoch % 10 == 9:
            checkpoint = {'model': model.state_dict(), 'optihmizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(
                checkpoint, f'model/checkpoint_{epoch}_loss{round(float(loss), 3)}_acc_{round(float(acc), 3)}.pth')
