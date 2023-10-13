import numpy as np


class LogLinUnitGenerator:
    """Simulates neural data based on movements using a log-linear relationship, as in Trucollo et al. 2008"""
    def __init__(self, num_chans, num_dof, pos_mult=1, vel_mult=1, noise_level=0.1):
        self.num_chans = num_chans
        self.num_dof = num_dof
        self.noise_level = noise_level  # std multiplier of the gaussian noise
        self.scaler = 1

        # create gaussian random relationships, using [P, V] as the latent state
        self.rand_mat = np.random.uniform(-1, 1, size=(2 * num_dof + 1, num_chans))

        # scale P/V (for example, you could place more emphasis on position by increasing pos_mult)
        mult = np.concatenate([np.repeat([float(pos_mult), float(vel_mult)], num_dof), [1]])
        mult = np.tile(mult.reshape(-1, 1), (1, num_chans))
        self.rand_mat *= mult

    def generate(self, pos, vel):
        """
        Generate neural data for each timestep given matrices of position and velocity.

        Parameters:
        - pos: np.array of shape (time_steps, num_dof) representing position at each timestep.
        - vel: np.array of shape (time_steps, num_dof) representing velocity at each timestep.

        Returns:
        - np.array of shape (time_steps, num_chans) representing neural activity at each timestep.
        """
        if pos.ndim == 1:
            pos = pos.reshape((1, -1))
        if vel.ndim == 1:
            vel = vel.reshape((1, -1))

        # Concatenate pos, vel, and a column of ones
        time_steps = pos.shape[0]
        state = np.hstack([pos - 0.5, vel, np.ones((time_steps, 1))])

        # Compute average firing rate
        avgfr = np.exp(self.scaler * state @ self.rand_mat)

        # Generate neural activity with Gaussian noise
        neural_activity = np.random.normal(loc=avgfr, scale=np.abs(avgfr * self.noise_level))

        return neural_activity
