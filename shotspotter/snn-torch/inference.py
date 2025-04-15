import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from common import read_spikes_from_disk, SNN, SpikesDataset, ConvLSTMSNN
import time
from power import watt_now
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-t', '--type', choices=['fc', 'conv'], required=True)
parser.add_argument('-s', '--set', choices=['train', 'test'], required=True)
parser.add_argument('-b', '--beta', required=True)
parser.add_argument('-p', '--power', action='store_true')

args = parser.parse_args()

train_data, train_labels, _, _, _, _, _, test_data, test_labels = read_spikes_from_disk(args.dataset)

BETA = float(args.beta)

if args.set == 'train':
    test_data = train_data
    test_labels = train_labels

if args.type == 'fc':
    snn = SNN(test_data.shape[2], BETA, test_data.shape[0]).to(device)
else:
    snn = ConvLSTMSNN(BETA, torch.tensor(0.17, dtype=torch.float32, requires_grad=True)).to(device)

snn.load_state_dict(torch.load(args.model, weights_only=True, map_location='cuda:0'))
snn.eval()

ds = SpikesDataset(test_data, torch.tensor(test_labels), False if args.type == 'fc' else True)
loader = DataLoader(ds, batch_size=100, shuffle=True)

# thread for power stuff
stop = threading.Event()
def record_consumption():
    while not stop.is_set():
        with open(args.out_path, 'a') as f:
            f.write(f'{watt_now()}\n')

        time.sleep(3)

thr = threading.Thread(target=record_consumption)

if args.power: thr.start()

print('Starting inference...')

t0 = time.time()
wrong_indexes = []
fp, fn = 0, 0
tp, tn = 0, 0
with torch.no_grad():
    correct = 0
    for data, labels, index in loader:
        data = data.to(device)
        labels = labels.to(device)
        index = index.to(device)
        
        if args.type == 'fc':
            data = data.permute(1, 0, 2)

        spk_rec, mem_rec = snn(data)
        
        # accuracy
        v, ind = spk_rec.sum(dim=0).max(1)
        correct += (labels == ind).sum()

        # fp and fn
        fp += ((labels != ind) & (labels == 0)).sum()
        fn += ((labels != ind) & (labels == 1)).sum()
        tp += ((labels == ind) & (labels == 1)).sum()
        tn += ((labels == ind) & (labels == 0)).sum()

        # get indexes of incorrect classifications
        for w in index[labels != ind]:
            wrong_indexes.append(int(w))
        
        # debug stuff
        total_spikes = spk_rec.sum(dim=0)

        #k = 0
        #for total in total_spikes:
        #    print(f'{total[0].item()}, {total[1].item()} - actual: {labels[k]}', 'WRONG' if ind[k] != labels[k] else '')
        #    k += 1

stop.set()
if args.power: thr.join()

print(f'Indexes of wrong samples', wrong_indexes)
print(f'Inference took: {time.time()-t0:.2f} seconds')
print(f'Accuracy: {100*(correct/len(ds)):.2f} ({correct}/{len(ds)})')
print(f'FN: {fn}, FP: {fp} (of {len(ds)-correct} incorrect classifications)')
print(f'TN: {tn}, TP: {tp} (of {correct} correct classifications)')
print(f'\nTrainable parameter count: {sum(p.numel() for p in snn.parameters())}')
