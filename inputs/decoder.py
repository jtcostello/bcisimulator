import numpy as np
import torch


class RealTimeDecoder:
    """
    Wrapper for decoders to allow realtime decoding.
    Simulates neural activity based on the current pos/vel, runs the decoder,
    and updates position by integrating velocity.

    The `model` should output both positions and velocities.
    """
    def __init__(self, num_dof, model, neuralsim, neural_scaler, output_scaler):
        #
        self.num_dof = num_dof
        self.model = model
        self.model.enable_online(True)
        self.neuralsim = neuralsim
        self.neural_scaler = neural_scaler
        self.output_scaler = output_scaler
        self.prev_desired_pos = 0.5 * np.ones((num_dof,))
        self.prev_actual_pos = 0.5 * np.ones((num_dof,))
        self.recent_neural = None

    def decode(self, desired_pos):
        # TODO: make sure this is correct, and how the cursor input works
        # generate neural data
        desired_vel = desired_pos - self.prev_desired_pos
        neural = self.neuralsim.generate(pos=desired_pos, vel=desired_vel)
        neural = self.neural_scaler.transform(neural.reshape(1, -1))
        self.recent_neural = neural.reshape(-1)
        self.prev_desired_pos = desired_pos

        # decode
        neural_tensor = torch.Tensor(neural).reshape((1, 1, -1))    # shape (batch_size, seq_len, num_chans)
        decoded_posvel = self.model(neural_tensor).reshape(-1)
        decoded_posvel = self.output_scaler.inverse_transform(decoded_posvel.reshape(1, -1)).reshape(-1)
        pos = decoded_posvel[:self.num_dof]
        vel = decoded_posvel[self.num_dof:]

        # integrate velocity
        beta = 0.8
        new_pos = beta * (self.prev_actual_pos + vel) + (1 - beta) * pos
        # new_pos = self.prev_actual_pos + vel
        # new_pos = pos
        new_pos = np.clip(new_pos, 0, 1)
        self.prev_actual_pos = new_pos
        return new_pos

    def get_recent_neural(self):
        return self.recent_neural
