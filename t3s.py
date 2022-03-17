from logging import raiseExceptions
from threading import Timer
import torch
from torch import nn
import numpy as np
import json
import random
import utils


timer = utils.Timer()

class MetricLearningDataset(torch.utils.data.Dataset):
    def __init__(self, file_train, grid2idx, metric="edr", max_len=128, triplet_num=10, device="cpu"):
        self.metric = metric
        self.triplet_num = triplet_num
        self.device = device
        self.train_dict = json.load(open(file_train))
        '''
        train_dict["trajs] : list of list of idx
        train_dict["sorted_index] : sorted matrix of most similarity trajs
        train_dict["origin_trajs"] : list of list of (lon, lat)
        train_dict["dis_matrix"] : matrix of distance
        '''
        
        self.data = []
        # turn the trajectory to max_len
        for traj in self.train_dict["trajs"]:
            traj = np.array(traj)
            if len(traj) > max_len:
                idx = [0] + sorted(random.sample(range(1, len(traj) - 1), max_len - 2)) + [len(traj) - 1]
                traj = traj[idx]
            elif len(traj) < max_len:
                traj = np.pad(traj, (0, max_len - len(traj)), "constant", constant_values=len(grid2idx))
            self.data.append(traj)
        self.data = torch.tensor(np.array(self.data), dtype=torch.long).to(device)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        positive = self.data[self.train_dict['sorted_index'][idx][:self.triplet_num]]
        negative = self.data[self.train_dict['sorted_index'][idx][-self.triplet_num:]]
        return anchor, positive, negative

class T3S(nn.Module):
    def __init__(
            self, vocab_size, dim_emb=256, head_num=8, encoder_layer_num=1, lstm_layer_num=1, pretrained_embedding=None
    ):
        super(T3S, self).__init__()
        self.lamb = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # lambda
        nn.init.constant_(self.lamb, 0.5)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # beta for RBF
        nn.init.constant_(self.beta, 0.5)
        if pretrained_embedding:
            self.embedding = nn.Embedding(vocab_size, dim_emb).from_pretrained(pretrained_embedding)
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
        tri_num = positive.shape[1]
        positive = positive.reshape(-1, positive.shape[2])
        negative = negative.reshape(-1, negative.shape[2])
        output_a = self.forward(anchor).unsqueeze(1).repeat(1, tri_num, 1)  # [bsz, triplet_num, emb]
        output_p = self.forward(positive).reshape(batch_size, tri_num, -1)  # [bsz, triplet_num, emb]
        output_n = self.forward(negative).reshape(batch_size, tri_num, -1)  # [bsz, triplet_num, emb]
        sim_pos = torch.exp(-torch.norm(output_a - output_p, p=2, dim=2) / (2 * self.beta))  # [bsz, triplet_num])
        sim_neg = torch.exp(-torch.norm(output_a - output_n, p=2, dim=2) / (2 * self.beta))  # [bsz, triplet_num])
        weight_pos = torch.softmax(torch.ones(tri_num) / torch.arange(1, tri_num + 1).float(), dim=0).to(sim_pos.device)  # [triplet_num]
        sim_pos = torch.einsum("bt, t -> b", sim_pos, weight_pos)  # [bsz]
        # sim_pos = sim_pos.sum(dim=1)
        sim_neg = sim_neg.mean(dim=1)  # [bsz]
        loss = 1 - sim_pos + sim_neg
        return torch.mean(loss)

    def pair_similarity(self, trajs):
        out = self.forward(trajs) # [bsz, emb]
        norm = torch.norm(out.unsqueeze(1)-out, dim=2, p=2) # [bsz, bsz]
        sim_matrix = torch.exp(-norm/(2*self.beta))
        return sim_matrix
        

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
                accs.append(len(np.intersect1d(sorted_index[i][:dataset.triplet_num], dataset[i][1].cpu().numpy())) / dataset.triplet_num)
        loss = np.mean(losses)
        acc = np.mean(accs)
        self.train()
        return loss, acc


    
def train_t3s(args):
    timer.tik("prepare data")
    
    # load args
    grid2idx_file = args.grid2idx
    train_dataset = args.train_dataset
    validate_dataset = args.validate_dataset
    batch_size = args.batch_size
    pretrained_embedding_file = args.pretrained_embedding
    emb_size = args.embedding_size
    learning_rate = args.learning_rate
    epochs = args.epoch_num
    checkpoint = args.checkpoint
    vp = args.visdom_port
    
    # prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    str_grid2idx = json.load(open(grid2idx_file))
    grid2idx = {eval(c): str_grid2idx[c] for c in list(str_grid2idx)}
    dataset = MetricLearningDataset(train_dataset, grid2idx, device=device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MetricLearningDataset(validate_dataset, grid2idx, device=device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # init model
    pre_emb = None
    if pretrained_embedding_file:
        pre_emb = np.load(pretrained_embedding_file)
    model = T3S(vocab_size=len(grid2idx) + 1, dim_emb=emb_size, pretrained_embedding=pre_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
    
    model.to(device)
    model.train()
    timer.tik("train")
    batch_count = 0
    train_loss_list = []
    validate_loss_list = [0]
    acc_list = [0]
    for epoch in range(epoch_start, epochs):
        # train
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            loss = model.calculate_loss(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timer.tok(f"epoch:{epoch} batch:{batch_idx} train loss:{loss.item():.4f}")
            train_loss_list.append(float(loss))
            batch_count += 1
            if vp != 0:
                env.line(
                    X=list(range(batch_count)),
                    Y=train_loss_list,
                    win=pane1
                )
                env.line(
                    X=list(range(epoch+1)),
                    Y=validate_loss_list,
                    win=pane2
                )
                env.line(
                    X=list(range(epoch+1)),
                    Y=acc_list,
                    win=pane3
                )
        if epoch % 1 == 0:
            validate_loss, acc = model.evaluate(test_loader, device)
            validate_loss_list.append(validate_loss)
            acc_list.append(acc)
            timer.tok(f"epoch:{epoch} batch:{batch_idx} validate loss:{validate_loss:.4f} acc:{acc:.4f}")
        if epoch % 10 == 1:
            checkpoint = {'model': model.state_dict(), 'optihmizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, f'model/checkpoint_{epoch}_loss{round(float(loss), 3)}_acc_{round(float(acc), 3)}.pth')
