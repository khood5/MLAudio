import argparse
import torch
from ss_dataset import MosaicDataset
from torch.utils.data import DataLoader
import time

from common import RGB_MEAN, RGB_STD, DEVICE, model_18, model_152

BATCH_SIZE=32

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', choices=['18', '152'], required=True)
parser.add_argument('-m', '--model_path', required=True)
args = parser.parse_args()

testing_set = MosaicDataset("data/dataset/testing.npz", 'testing', rgb_mean=RGB_MEAN, rgb_std=RGB_STD)
testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE, shuffle=True)

if args.type == '18':
    model = model_18
else:
    model = model_152

# map from cuda to cpu (since we saved from cuda state) to infer on raspberry pi
model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=torch.device('cpu')))
model = model.to(DEVICE)
model.eval() # disables batch norm

t = time.time()

correct = 0
total = 0
with torch.no_grad():
    for data in testing_loader:
        mosaics, labels = data

        if torch.cuda.is_available():
            mosaics = mosaics.to('cuda')
            labels = labels.to('cuda')

        outputs = model(mosaics)
        v, ind = torch.max(outputs, 1)
        correct += torch.sum(ind == labels).item()
        total += BATCH_SIZE

t1 = time.time()

print(f'Accuracy on test set: {100 * correct / total} %')
print(f'Total inference time: {t1-t:.3f}')