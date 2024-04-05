import torch
from torchvision import transforms
from audioDataLoader import audioDataloader
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process train, valid, and test datasets.')
parser.add_argument('train_dataset', type=str, help='Path to the train dataset file')
parser.add_argument('valid_dataset', type=str, help='Path to the valid dataset file')
parser.add_argument('test_dataset', type=str, help='Path to the test dataset file')
args = parser.parse_args()

train_dataset = args.train_dataset
valid_dataset = args.valid_dataset
test_dataset = args.test_dataset

print("Using train dataset:", train_dataset)
print("Using valid dataset:", valid_dataset)
print("Using test dataset:", test_dataset)

data_transform = transforms.Compose([
        transforms.Normalize(mean=[2.3009], std=[42.1936]) 
    ])

train_data = audioDataloader(index_file=train_dataset, transforms=data_transform)
valid_data = audioDataloader(index_file=valid_dataset, transforms=data_transform)
test_data = audioDataloader(index_file=test_dataset, transforms=data_transform)

features_dim = len(train_data[0][0][0])
timestep_dim = len(train_data[0][0][0][0])

print("making training numpy file")
data_instance_dim = len(train_data)
neuroTrain = torch.zeros((data_instance_dim, features_dim, timestep_dim))
for i in tqdm(range(data_instance_dim)):
    neuroTrain[i] = train_data[i][0]
neuroTrain = neuroTrain.numpy
np.save('neuroTrain.npy', neuroTrain)

print("making validation numpy file")
data_instance_dim = len(valid_data)
neuroValid = torch.zeros((data_instance_dim, features_dim, timestep_dim))
for i in tqdm(range(data_instance_dim)):
    neuroValid[i] = valid_data[i][0]
neuroValid = neuroValid.numpy
np.save('neuroValid.npy', neuroValid)

print("Making testing numpy file")
data_instance_dim = len(test_data)
neuroTest = torch.zeros((data_instance_dim, features_dim, timestep_dim))
for i in tqdm(range(data_instance_dim)):
    neuroTest[i] = test_data[i][0]
neuroTest = neuroTest.numpy
np.save('neuroTest.npy', neuroTest)