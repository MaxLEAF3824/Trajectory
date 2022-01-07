import torch
from torch import nn

m = 800
n = 800
dim_emb = 512
dim_ff = 2048


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_cell = nn.LSTM()

    def forward(self):
        pass


class T3S(nn.Module):
    def __init__(self):
        super(T3S, self).__init__()
        self.encoder = Encoder().cuda()
        self.lstm = LSTM().cuda()

    def forward(self):
        pass
