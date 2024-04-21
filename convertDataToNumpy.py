import torch
from torchvision import transforms
from audioDataLoader import audioDataloader
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process train, valid, and test datasets.')
parser.add_argument('train_dataset', type=str, help='Path to the train dataset file')
parser.add_argument('-v', '--valid_dataset', default='', type=str, help='Path to the valid dataset file')
parser.add_argument('-t', '--test_dataset', default='', type=str, help='Path to the test dataset file')
parser.add_argument('--train_out', default='neuroTrain', type=str, help='Path to the train output file')
parser.add_argument('--valid_out', default='neuroValid', type=str, help='Path to the valid output file')
parser.add_argument('--test_out', default='neuroTest', type=str, help='Path to the test output file')
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

timestep_dim = len(train_data[0][0][0][0])
features_dim = len(train_data[0][0][0])
print(f"Number of timestep: {timestep_dim}")
print(f"Number of features: {features_dim}")

print("making training numpy file")
data_instance_dim = len(train_data)
print(f"Number of data instances: {data_instance_dim}")
neuroTrain = torch.zeros((data_instance_dim, features_dim, timestep_dim))
neuroTrainLabels = torch.zeros((data_instance_dim))
for i in tqdm(range(data_instance_dim)):
    neuroTrainLabels[i] = train_data[i][1]
neuroTrain = neuroTrain.numpy()
np.save(args.train_out, neuroTrain)
np.save(f"{args.train_out}_labels", neuroTrainLabels)

if(valid_dataset != ''):
    valid_data = audioDataloader(index_file=valid_dataset, transforms=data_transform)
    print("making validation numpy file")
    data_instance_dim = len(valid_data)
    print(f"Number of data instances: {data_instance_dim}")
    neuroValid = torch.zeros((data_instance_dim, features_dim, timestep_dim))
    neuroValidLabels = torch.zeros((data_instance_dim))
    for i in tqdm(range(data_instance_dim)):
        neuroValidLabels[i] = valid_data[i][1]
    neuroValid = neuroValid.numpy()
    np.save(args.valid_out, neuroValid)
    np.save(f"{args.valid_out}_labels", neuroValidLabels)

if(test_dataset != ''):
    test_data = audioDataloader(index_file=test_dataset, transforms=data_transform)
    print("Making testing numpy file")
    data_instance_dim = len(test_data)
    print(f"Number of data instances: {data_instance_dim}")
    neuroTest = torch.zeros((data_instance_dim, features_dim, timestep_dim))
    neuroTestLabels = torch.zeros((data_instance_dim))
    for i in tqdm(range(data_instance_dim)):
        neuroTestLabels[i] = test_data[i][1]
    neuroTest = neuroTest.numpy()
    np.save(args.test_out, neuroTest)
    np.save(f"{args.test_out}_labels", neuroTestLabels)