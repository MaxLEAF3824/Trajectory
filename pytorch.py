import torch
from torch import nn

class MetricLearningDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, metric):
        self.X = X
        self.Y = Y
        self.metric = metric

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.metric[idx]

class T3S(nn.Module):
    def __init__(self,vocab_size, dim_emb=256, head_num=8, layer_num=6, num_layers=1,pretrain_embedding=None):
        super(T3S, self).__init__()
        self.lamb = torch.rand(1) # lambda
        if pretrain_embedding:
            self.embedding = nn.Embedding(vocab_size, dim_emb).from_pretrained(pretrain_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_emb, head_num)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_num)
        self.lstm = nn.LSTM(dim_emb, dim_emb, num_layers, batch_first=True)

    def forward(self, x):
        '''
        x: [batch_size, seq_len]
        '''
        emb_x = self.embedding(x) # emb_x: [batch_size, seq_len, dim_emb]
        encoder_out = self.encoder(emb_x) # encoder_out: [batch_size, seq_len, dim_emb]
        encoder_out = torch.mean(encoder_out, 1) # encoder_out: [batch_size, dim_emb]
        lstm_out, (h_n, c_n) = self.lstm(emb_x) # lstm_out: [batch_size, seq_len, dim_emb]
        lstm_out = lstm_out[:, -1, :] # lstm_out: [batch_size, dim_emb]
        output = self.lamb * encoder_out + (1 - self.lamb) * lstm_out # output: [batch_size, dim_emb]
        return output

def Trainer(model, train_loader, test_loader, loss_func, optimizer, epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y, metric) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            metric = metric.to(device)
            optimizer.zero_grad()
            output = model(x)

if __name__ == "__main__":
    src = torch.tensor([1,2,4,5,6,7,4])
    src.unsqueeze_(0)
    T3S = T3S(vocab_size=100).cuda()
    output = T3S(src)
    print(output.shape)    