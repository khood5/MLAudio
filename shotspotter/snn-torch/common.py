import numpy as np
import torch.nn as nn
import torch
import snntorch as snn
from torch.utils.data import Dataset, DataLoader

def read_spikes_from_disk(path):
    data = np.load(path)

    # compatability with old datasets
    val_filenames = []
    if 'validation_filenames' in data:
        val_filenames = data['validation_filenames']

    return data['train_set'], data['train_labels'], data['train_gunshot_data'], data['validation_set'], data['validation_labels'], data['validation_gunshot_data'], val_filenames, data['test_set'], data['test_labels']

def write_spikes_to_disk(path, metadata, train, train_labels, train_gunshot_data, val, val_labels, val_gunshot_data, val_filenames, test, test_labels):
    np.savez(path, metadata=metadata, train_set=train, validation_set=val, test_set=test, train_labels=train_labels,
            validation_labels=val_labels, test_labels=test_labels, train_gunshot_data=train_gunshot_data,
            validation_gunshot_data=val_gunshot_data, validation_filenames=val_filenames)

class SNN(nn.Module):
    def __init__(self, input_neurons, beta, num_timesteps):
        super().__init__()

        self.num_timesteps = num_timesteps

        self.fc1 = nn.Linear(input_neurons, 800)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(800, 800)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(800, 2)
        self.lif3 = snn.Leaky(beta=beta)

    # x will be (timestep x batch x neuron) shape
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spikes = []
        mem_rec = []

        # Note: one timestep is a full pass
        for step in range(self.num_timesteps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spikes.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spikes, dim=0), torch.stack(mem_rec, dim=0)

# see https://snntorch.readthedocs.io/en/latest/snn.neurons_sconvlstm.html
class ConvLSTMSNN(nn.Module):
    def __init__(self, beta, threshold):
        super().__init__()

        in_channels = 1
        out_channels = 32
        kernel_size = 3
        max_pool = 2
        avg_pool = 2
        flattened_input = 1536
        num_outputs = 2
        beta = 0.9

        spike_grad_lstm = snn.surrogate.straight_through_estimator()
        spike_grad_fc = snn.surrogate.fast_sigmoid(slope=5)

        # initialize layers
        self.sclstm1 = snn.SConv2dLSTM(
            1,
            16,
            5,
            max_pool=max_pool,
            spike_grad=spike_grad_lstm,
            threshold=threshold,
            learn_threshold=False
        )
        self.sclstm2 = snn.SConv2dLSTM(
            16,
            32,
            3,
            avg_pool=avg_pool,
            spike_grad=spike_grad_lstm,
            threshold=threshold,
            learn_threshold=False
        )
        self.sclstm3 = snn.SConv2dLSTM(
            32,
            64,
            3,
            avg_pool=avg_pool,
            spike_grad=spike_grad_lstm,
            threshold=threshold,
            learn_threshold=False
        )
        self.fc1 = nn.Linear(flattened_input, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)
        self.fc2 = nn.Linear(128, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.sclstm1.reset_mem()
        syn2, mem2 = self.sclstm2.reset_mem()
        syn3, mem3 = self.sclstm3.reset_mem()
        mem4 = self.lif1.init_leaky()
        mem5 = self.lif2.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []

        # Number of steps assuming x is [N, T, C, H, W] with
        # N = Batches, T = Time steps, C = Channels,
        # H = Height, W = Width
        num_steps = x.size()[1]

        for step in range(num_steps):
            x_step = x[:, step, :, :, :]
            spk1, syn1, mem1 = self.sclstm1(x_step, syn1, mem1)
            spk2, syn2, mem2 = self.sclstm2(spk1, syn2, mem2)
            spk3, syn3, mem3 = self.sclstm3(spk2, syn3, mem3)
            cur = self.fc1(spk3.flatten(1))
            spk4, mem4 = self.lif1(cur, mem4)
            cur2 = self.fc2(spk4)
            spk5, mem5 = self.lif2(cur2, mem5)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)

# dataset for batches
class SpikesDataset(Dataset):
    def __init__(self, data, labels, conv=False):
        self.data = data
        self.labels = labels
        self.isConv = conv

    def __len__(self):
        if self.isConv == False:
            return self.data.shape[1]
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.isConv == False:
            return self.data[:, index, :], self.labels[index], index
        return self.data[index, :, :, :, :], self.labels[index], index