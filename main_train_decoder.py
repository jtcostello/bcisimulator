import torch.optim
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import argparse

import neuralsim
import decoders.rnn
import decoders.ridge
import data_loading as data_loading


# parse training options
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--num_chans', type=int, default=100)
parser.add_argument('-n', '--neural_noise_level', type=float, default=0.3)
parser.add_argument('--num_dof', type=int, default=2)
parser.add_argument('-d', '--dataset', type=str, default="dataset_20230929_2158.pkl")
parser.add_argument('--decoder_type', type=str, default='rnn')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--train_data_frac', type=float, default=0.8)
parser.add_argument('--seq_len', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_plot', action='store_false')
parser.add_argument('--no_save', action='store_false')
args = parser.parse_args()
num_chans = args.num_chans
neural_noise_level = args.neural_noise_level
num_dof = args.num_dof
dataset_fname = args.dataset
decoder_type = args.decoder_type
epochs = args.epochs
train_data_frac = args.train_data_frac
seq_len = args.seq_len
batch_size = args.batch_size
plot_training = args.no_plot
save_decoder = args.no_save


# load movement data
with open(os.path.join("data", "movedata", dataset_fname), 'rb') as f:
    df = pickle.load(f)
pos = np.stack(df.current_position.to_numpy())  # shape (num_timepts, num_dof)
vel = np.vstack((np.zeros_like((1, pos.shape[1])), pos[1:, :] - pos[:-1, :]))
posvel = np.hstack((pos, vel))
num_trials = df.trial_number.iloc[-1] - df.trial_number.iloc[0]
num_secs = (df.timestep.iloc[-1] - df.timestep.iloc[0]) / 1000
print(f"Loaded {num_trials} trials, with {num_secs:.1f} seconds of data")
print(f"Number of samples: {posvel.shape[0]}")

# generate fake neural data from the movements
neuralsim = neuralsim.LogLinUnitGenerator(num_chans, num_dof, pos_mult=0.5, vel_mult=1, noise_level=neural_noise_level)
neural = []
for i in range(len(pos)):
    neural.append(neuralsim.generate(pos=pos[i], vel=vel[i]))
neural = np.vstack(neural)      # shape (num_timepts, num_chans)

# split train/test
x_train, x_test, y_train, y_test = train_test_split(neural, posvel, train_size=train_data_frac, shuffle=False)

# normalize inputs & outputs
neural_scaler = StandardScaler()
neural_scaler.fit(x_train)
x_train_norm = neural_scaler.transform(x_train)
x_test_norm = neural_scaler.transform(x_test)
output_scaler = StandardScaler()
output_scaler.fit(y_train)
y_train_norm = output_scaler.transform(y_train)
y_test_norm = output_scaler.transform(y_test)

# add time history (results in tensor of shape (num_samples, num_chans, seq_len))
x_train_norm_hist = data_loading.add_time_history(x_train_norm, seq_len=seq_len)
x_test_norm_hist = data_loading.add_time_history(x_test_norm, seq_len=seq_len)

# setup dataloaders
dataset_train = data_loading.SequenceDataset(x_train_norm_hist, torch.tensor(y_train_norm))
dataset_test = data_loading.SequenceDataset(x_test_norm_hist, torch.tensor(y_test_norm))
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, drop_last=False)

# create and train decoder
num_outputs = 2 * num_dof   # both pos & vel for each dof
if decoder_type == 'ridge':
    # TODO: finish RR training
    pass

elif decoder_type == 'rnn':
    # setup model and optimizer
    lr = 5e-4
    weight_decay = 0.001
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    model = decoders.rnn.RNN(num_chans, num_outputs, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    # train & evaluate accuracy
    loss_history = model.fit(loader_train, optimizer, loss_fn, epochs, verbose=True)
    y, yhat, _, _ = model.eval_perf(loader_test)
    y = output_scaler.inverse_transform(y)
    yhat = output_scaler.inverse_transform(yhat)

if plot_training:
    plt.plot(loss_history)
    plt.title("Training Loss by Epoch")
    plt.show()
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    titles = ["x pos", "y pos", "x vel", "y vel"]
    for i in range(4):
        axs[i].plot(y[:, i], label="ground truth")
        axs[i].plot(yhat[:, i], label="predicted")
        axs[i].set_title(titles[i])
        axs[i].legend()
    plt.tight_layout()
    plt.show()


# save decoder to file
if save_decoder:
    fname = f"decoder_{decoder_type}_{dataset_fname}"
    with open(os.path.join("data", "trained_decoders", fname), 'wb') as f:
        pickle.dump((model, neuralsim, neural_scaler, output_scaler), f)
    print(f"Saved decoder to {fname}")


