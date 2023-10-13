import torch
import torch.nn as nn
import numpy as np


# default RNN training params
RNN_CONFIG = {
    "lr": 5e-4,
    "weight_decay": 0.001,
    "hidden_size": 512
}


class RNN(nn.Module):
    """A recurrent neural network decoder"""

    def __init__(self, num_inputs, num_outputs, hidden_size=512, num_layers=1, dropout=0.0, rnn_type='lstm', device='cpu'):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(num_inputs, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(num_inputs, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(num_inputs, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)
        self.hidden = None
        self.is_online = False

    def enable_online(self, is_online=True):
        # when we're running online, save the hidden state in memory rather than resetting it each batch
        self.hidden = self.init_hidden(1) if is_online else None
        self.is_online = is_online

    def forward(self, x):
        # x should have shape (batches, sequence length, features)
        x = x.to(self.device)
        # Pass through the rnn and linear layers:
        h = self.hidden if self.hidden else self.init_hidden(x.shape[0])
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1])  # out now has shape (batch_size, num_outs) like (64, 2)
        if self.is_online:
            self.hidden = h
            return out.cpu().detach().numpy()
        else:
            return out

    def init_hidden(self, batch_size):
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device)

    def fit(self, dataloader, optimizer, loss_fn, epochs, verbose=True):
        self.train()
        epoch_losses = []
        for i in range(epochs):
            losses = []
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                yhat = self.forward(x)
                loss = loss_fn(yhat, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_losses.append(np.mean(losses))
            if verbose:
                print(f'epoch {i}, training loss = {np.mean(losses)}')
        return epoch_losses

    def eval_perf(self, dataloader, verbose=True):
        self.eval()
        # get model predictions
        all_y, all_yhat = [], []
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.forward(x)
            all_y.append(y.cpu().detach().numpy())
            all_yhat.append(yhat.cpu().detach().numpy())
        all_y = np.array(all_y).squeeze(0)
        all_yhat = np.array(all_yhat).squeeze(0)

        # calc performance
        mse = (np.square(all_y - all_yhat)).mean()
        corr = np.diag(np.corrcoef(all_y, all_yhat, rowvar=False)[:y.shape[1], y.shape[1]:])
        if verbose:
            print(f'avg correlation = {corr.mean()}, mse = {mse}')
        return all_y, all_yhat, mse, corr
