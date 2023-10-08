import numpy as np


class LogLinUnitGenerator:
    """Simulates neural data based on movements using a log-linear relationship, as in Trucollo et al. 2008"""
    def __init__(self, num_chans, num_dof, pos_mult=1, vel_mult=1, noise_level=0.1):
        self.num_chans = num_chans
        self.num_dof = num_dof
        self.std_multiplier = noise_level   # std multiplier of the gaussian noise
        self.scaler = 1

        # create gaussian random relationships, using [P, V] as the latent state
        self.rand_mat = np.random.uniform(-1, 1, size=(2*num_dof + 1, num_chans))

        # scale P/V (for example, you could place more emphasis on position by increasing pos_mult)
        mult = np.concatenate([np.repeat([float(pos_mult), float(vel_mult)], num_dof), [1]])
        mult = np.tile(mult.reshape(-1, 1), (1, num_chans))
        self.rand_mat *= mult

    def generate(self, pos, vel):
        # TODO: make this work with matrices
        # [1 p1 p2 v1 v2] * rand = [ch1 ch2 ch3 ch4 ...]
        # (1, 2*num_dof+1) * (2*num_dof+1, num_units) = (1, num_units)
        state = np.concatenate([pos-0.5, vel, [1]], axis=0).reshape((1, -1))
        avgfr = np.exp(self.scaler * state @ self.rand_mat).reshape((-1,))

        # use a normal distribution to generate random firing rates - mean and var are equal
        return np.random.normal(loc=avgfr, scale=np.abs(avgfr*self.std_multiplier))
