import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from common import read_spikes_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# things to consider
# maybe delta mod threshold is still too large, spiking data looks very noisy
# figure out what is wrong with validation calc

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True, help='path for log file and model output')
parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-e', '--epochs', required=True)
parser.add_argument('-r', '--learn_rate', required=True)
parser.add_argument('-b', '--batch_size', default=12)

args = parser.parse_args()

OUT_PATH = args.output+'/' if args.output[-1] != '/' else args.output
EPOCHS = int(args.epochs)
LEARN_RATE = float(args.learn_rate)
BATCH_SIZE = int(args.batch_size)

# log settings to output
with open(OUT_PATH+'details.txt', 'a') as f:
    f.write(f'Dataset filename: {args.dataset}\n')
    f.write(f'Learning Rate: {args.learn_rate}\n')
    f.write(f'Batch size: {args.batch_size}\n')

# load and preprocess dataset
training_data, training_labels, _, validation_data, validation_labels, _, _, test_data, test_labels = read_spikes_from_disk(args.dataset)

INPUT_NEURON_COUNT = training_data.shape[2]

# temporary function just to convert data from s2s format from make_data.py to snntorch format
# def parse_s2s(d):
#     all_samples = []
#     for sample in d:
#         channels = []
#         for channel in sample:
#             channels.append([])
#             for timestep in channel:
#                 channels[-1].append(timestep[2])
#         all_samples.append(channels)

#     return torch.tensor(all_samples, dtype=torch.float32).permute(2, 0, 1)

# dm_train_data = parse_s2s(training_data)
# dm_val_data = parse_s2s(validation_data)

beta = 0.9 # slow decay
num_timesteps = training_data.shape[0]

class SNN(nn.Module):
    def __init__(self, input_neurons):
        super().__init__()
        self.fc1 = nn.Linear(input_neurons, 150)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(150, 150)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(150, 2)
        self.lif3 = snn.Leaky(beta=beta)

    # x will be (timestep x batch x neuron) shape
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spikes = []
        mem_rec = []

        # Note: one timestep is a full pass
        for step in range(num_timesteps):
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


# data is a row to be appended to the csv of format (train_loss, val_loss, val_accuracy) 
def append_log(filename, data):
    with open(OUT_PATH+filename, 'a') as f:
        f.write(f'{data[0]},{data[1]},{data[2]}\n')


ds = SpikesDataset(training_data, torch.tensor(training_labels))
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

ds_val = SpikesDataset(validation_data, torch.tensor(validation_labels))
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)

net = SNN(INPUT_NEURON_COUNT).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE, betas=(0.9, 0.999))

train_loss_log = []
val_loss_log = []
val_accuracy_log = []
for i in range(EPOCHS):
    print(f'Starting epoch {i}.')
    t0 = time.time()
        
    # train set run and update
    net.train()
    epoch_losses = []
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        
        data = data.permute(1, 0, 2) # permute so we have right shape
        
        spk_rec, mem_rec = net(data)

        # interesting part, we are doing cross entropy loss per timestep here
        loss_val = torch.zeros((1), dtype=torch.float64, device=device)
        for s in range(num_timesteps):
            loss_val += loss(mem_rec[s], labels)
            

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_losses.append(loss_val.item())

    print(f'Train loss: {sum(epoch_losses)/len(epoch_losses):.2f}')
    train_loss_log.append(sum(epoch_losses)/len(epoch_losses))

    # validation set
    epoch_losses = []
    with torch.no_grad():
        net.eval()

        correct = 0
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            data = data.permute(1, 0, 2)

            spk_rec, mem_rec = net(data)
            
            # accuracy
            v, ind = spk_rec.sum(dim=0).max(1)
            correct += (labels == ind).sum()

            loss_val = torch.zeros((1), dtype=torch.float64, device=device)
            for s in range(num_timesteps):
                loss_val += loss(mem_rec[s], labels)

            epoch_losses.append(loss_val.item())

        print(f'Val loss: {sum(epoch_losses)/len(epoch_losses):.2f}')
        val_loss_log.append(sum(epoch_losses)/len(epoch_losses))
        val_accuracy_log.append((correct/len(ds_val))*100)
        print(f'Val accuracy: {val_accuracy_log[-1]:.2f}')

        # write logs and stuff to disk
        append_log('log.csv', (train_loss_log[-1], val_loss_log[-1], val_accuracy_log[-1]))

        if len(val_accuracy_log) == 1 or val_accuracy_log[-1] > max(val_accuracy_log[:-1]):
            print('New best validation accuracy, writing to disk...')
            torch.save(net.state_dict(), OUT_PATH+'model.pth')

        print(f'Took {time.time()-t0:.2f} seconds')
        print('-----')