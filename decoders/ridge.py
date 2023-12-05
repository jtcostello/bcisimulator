import numpy as np
import torch
import torch.nn as nn


class RidgeRegression(nn.Module):
    """A ridge regression decoder"""

    def __init__(self, num_inputs, num_outputs, lmbda=0.1):
        super().__init__()
        self.num_inputs = num_inputs    # equal to num_features * seq_len(history)
        self.num_outputs = num_outputs
        self.lmbda = lmbda
        self.weights = None

    def enable_online(self, is_online=True):
        pass

    def forward(self, x):
        # x should have shape (batches, sequence length, features)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = x.reshape(-1, self.num_inputs)
        y_pred = np.dot(x, self.weights)
        return y_pred

    def fit(self, x, y):
        # x should have shape (batches, sequence length, features)
        # y should have shape (batches, out_features)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        x = x.reshape(-1, self.num_inputs)
        self.weights = np.dot(np.linalg.inv(np.dot(x.T, x) + self.lmbda * np.eye(self.num_inputs)),
                              np.dot(x.T, y))

    def eval_perf(self, x, y, verbose=True):
        # x should have shape (batches, sequence length, features)
        # y should have shape (batches, out_features)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        x = x.reshape(-1, self.num_inputs)
        y_pred = np.dot(x, self.weights)

        # calc performance
        mse = np.mean((y - y_pred) ** 2)
        corr = np.diag(np.corrcoef(y, y_pred, rowvar=False)[:y.shape[1], y.shape[1]:])
        if verbose:
            print(f'avg correlation = {corr.mean()}, mse = {mse}')
        return y, y_pred, mse, corr

