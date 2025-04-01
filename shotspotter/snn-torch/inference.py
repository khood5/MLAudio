import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from common import read_spikes_from_disk, SNN, SpikesDataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-m', '--model', required=True)

args = parser.parse_args()

train_data, train_labels, _, _, _, _, _, test_data, test_labels = read_spikes_from_disk(args.dataset)

#test_data = train_data
#test_labels = train_labels

snn = SNN(test_data.shape[2], 0.9, test_data.shape[0]).to(device)
snn.load_state_dict(torch.load(args.model, weights_only=True))
snn.eval()

ds = SpikesDataset(test_data, torch.tensor(test_labels))
loader = DataLoader(ds, batch_size=12, shuffle=True)

t0 = time.time()
with torch.no_grad():
    correct = 0
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        
        data = data.permute(1, 0, 2)

        spk_rec, mem_rec = snn(data)
        
        # accuracy
        v, ind = spk_rec.sum(dim=0).max(1)
        correct += (labels == ind).sum()

print(f'Inference took: {time.time()-t0:.2f} seconds')
print(f'Accuracy: {100*(correct/len(ds)):.2f}')
