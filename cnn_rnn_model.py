import torch
from torch import nn

class MetrixPair(nn.Module):
    def __init__(self):
        super(MetrixPair, self).__init__()
        # self.weight = nn.Parameter(torch.Tensor(128, 128).uniform_(-1/128., 1/128.),requires_grad = True)
        self.weight = nn.Parameter(torch.Tensor(1, 13).uniform_(-1/13., 1/13.), requires_grad = True) # for BiLSTM

    def forward(self, x):
        return x.mul(self.weight), self.weight

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.Siamese= MetrixPair()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=13,
            hidden_size=32,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = x.squeeze(1)
        x, weight = self.Siamese(x)
        x = x.unsqueeze(1)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out, weight

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.Siamese= MetrixPair()

        self.inp = nn.Linear(13, 64)
        self.hidden = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.Siamese(x)
        # x = x.unsqueeze(0)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        out = self.inp(x.view(-1,13))
        out = self.hidden(out)
        out = self.out(out)
        return out