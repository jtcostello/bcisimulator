import torch
from torch.utils.data import Dataset, DataLoader


def add_time_history(x, seq_len=3):
    """
    Adds time history to the input features
    Input shape (num_samples, num_chans)
    Output shape (num_samples, num_chans, seq_len)
    """
    xin = torch.tensor(x)

    # add time delays to input features
    xhist = torch.zeros((int(xin.shape[0]), int(xin.shape[1]), seq_len))
    xhist[:, :, 0] = xin
    for i in range(1, seq_len):
        xhist[i:, :, i] = xin[0:-i, :]

    # make the last timestep the most recent data
    xhist = torch.flip(xhist, (2,))

    # put in shape (batches, sequence length, features)
    xhist = xhist.permute(0, 2, 1)

    return xhist


class SequenceDataset(Dataset):
    """Simple dataset for sequences of data"""
    def __init__(self, x, y):
        self.x = x.to(torch.float)
        self.y = y.to(torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]