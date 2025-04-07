import snntorch as snn
import snntorch.functional as SF
from snntorch import spikegen
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from common import read_spikes_from_disk, SNN, SpikesDataset, ConvLSTMSNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

beta = 0.9 # slow decay
num_timesteps = training_data.shape[1]

# data is a row to be appended to the csv of format (train_loss, val_loss, val_accuracy) 
def append_log(filename, data):
    with open(OUT_PATH+filename, 'a') as f:
        f.write(f'{data[0]},{data[1]},{data[2]}\n')


ds = SpikesDataset(training_data, torch.tensor(training_labels), True)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

ds_val = SpikesDataset(validation_data, torch.tensor(validation_labels), True)
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)

net = ConvLSTMSNN(beta, 0.1).to(device)

loss = SF.ce_rate_loss()
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
    for data, labels, _ in loader:
        data = data.to(device)
        labels = labels.to(device)
        
        spk_rec, mem_rec = net(data)

        loss_val = loss(spk_rec, labels)

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
        for data, labels, _ in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            spk_rec, mem_rec = net(data)
            
            # accuracy
            v, ind = spk_rec.sum(dim=0).max(1)
            correct += (labels == ind).sum()

            loss_val = loss(spk_rec, labels)

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