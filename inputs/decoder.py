import numpy as np
import torch
from collections import deque


class RealTimeDecoder:
    """
    Wrapper for decoders to allow realtime decoding.
    Simulates neural activity based on the current pos/vel, runs the decoder,
    and updates position by integrating velocity.

    The `model` should output both positions and velocities (e.g. [pos1 pos2 vel1 vel2])
    """
    def __init__(self, num_dof, model, neuralsim, neural_scaler, output_scaler, seq_len, integration_beta=0.98):
        self.num_dof = num_dof
        self.model = model
        self.model.enable_online(True)
        self.neuralsim = neuralsim
        self.neural_scaler = neural_scaler
        self.output_scaler = output_scaler
        self.seq_len = seq_len
        self.integration_beta = integration_beta        # 0.9 means 90% of position is from integrated velocity
        print(f"Decoder using {integration_beta * 100:.1f}% integrated velocity, "
              f"{100 - integration_beta * 100:.1f}% decoded position")
        print(f"Neural simulator: {self.neuralsim.num_chans} chans with {self.neuralsim.noise_level} noise level")

        # setup neural history & init with zeros
        num_chans = neuralsim.num_chans
        self.neural_history = deque(maxlen=seq_len)
        for _ in range(seq_len):
            self.neural_history.append(np.zeros((num_chans,)))

        self.prev_desired_pos = 0.5 * np.ones((num_dof,))
        self.prev_actual_pos = 0.5 * np.ones((num_dof,))

    def decode(self, desired_pos):
        # generate neural data
        desired_vel = desired_pos - self.prev_desired_pos
        neural = self.neuralsim.generate(pos=desired_pos, vel=desired_vel)
        neural = self.neural_scaler.transform(neural.reshape(1, -1))
        self.neural_history.append(neural.reshape(-1))
        neural_history_np = np.array(self.neural_history)
        self.prev_desired_pos = desired_pos

        # decode (model expects shape (batch_size, seq_len, num_chans))
        neural_tensor = torch.Tensor(neural_history_np).reshape((1, self.seq_len, -1))
        decoded_posvel = self.model(neural_tensor).reshape(-1)
        decoded_posvel = self.output_scaler.inverse_transform(decoded_posvel.reshape(1, -1)).reshape(-1)
        pos = decoded_posvel[:self.num_dof]
        vel = decoded_posvel[self.num_dof:]

        # integrate velocity
        new_pos = self.integration_beta * (self.prev_actual_pos + vel) + (1 - self.integration_beta) * pos
        new_pos = np.clip(new_pos, 0, 1)
        self.prev_actual_pos = new_pos
        return new_pos

    def set_position(self, pos):
        self.prev_desired_pos = pos
        self.prev_actual_pos = pos

    def get_recent_neural(self):
        return self.neural_history[-1]
