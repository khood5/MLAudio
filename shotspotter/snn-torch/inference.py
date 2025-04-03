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

test_data = train_data
test_labels = train_labels

snn = SNN(test_data.shape[2], 0.9, test_data.shape[0]).to(device)
snn.load_state_dict(torch.load(args.model, weights_only=True, map_location='cuda:0'))
snn.eval()

ds = SpikesDataset(test_data, torch.tensor(test_labels))
loader = DataLoader(ds, batch_size=100, shuffle=True)

t0 = time.time()
wrong_indexes = []
fp, fn = 0, 0
with torch.no_grad():
    correct = 0
    for data, labels, index in loader:
        data = data.to(device)
        labels = labels.to(device)
        index = index.to(device)
        
        data = data.permute(1, 0, 2)

        spk_rec, mem_rec = snn(data)
        
        # accuracy
        v, ind = spk_rec.sum(dim=0).max(1)
        correct += (labels == ind).sum()

        # fp and fn
        fp += ((labels != ind) & (labels == 0)).sum()
        fn += ((labels != ind) & (labels == 1)).sum()

        # get indexes of incorrect classifications
        for w in index[labels != ind]:
            wrong_indexes.append(int(w))
    

print(f'Indexes of wrong samples', wrong_indexes)
print(f'Inference took: {time.time()-t0:.2f} seconds')
print(f'Accuracy: {100*(correct/len(ds)):.2f} ({correct}/{len(ds)})')
print(f'FN: {fn}, FP: {fp} (of {len(ds)-correct} incorrect classifications)')