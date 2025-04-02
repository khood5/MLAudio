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

        self.fc1 = nn.Linear(input_neurons, 100)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(100, 100)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(100, 2)
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


# dataset for batches
class SpikesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        return self.data[:, index, :], self.labels[index]
