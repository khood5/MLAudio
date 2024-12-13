import argparse
import torch
from ss_dataset import MosaicDataset
from torch.utils.data import DataLoader

from common import RGB_MEAN, RGB_STD, DEVICE, model_18

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
    pass #152

model.load_state_dict(torch.load(args.model_path, weights_only=True))
model = model.to(DEVICE)

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

print(f'accuracy: {100 * correct / total} %')