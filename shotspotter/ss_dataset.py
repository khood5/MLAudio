from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor, Normalize

# expects index csv where column 1 is file path and column 2 is class
def save_from_index(index_path, output_path):
    print(f"Loading dataset index file from {index_path}")

    data = []
    with open(index_path, 'r') as f:
        for l in f.readlines():
            path, label = l.replace('\n', '').split(',')
            image = np.asarray(Image.open(path))
            data.append([image, label])

        random.shuffle(data)
        training_size = int(len(data)*0.8)
        other_size = int(len(data)*0.1)

        training = data[0:training_size]
        np.savez(output_path+"/training.npz", images=[i[0] for i in training], labels=[int(i[1]) for i in training])
        print('Successfully written training.npz')

        validation = data[training_size:training_size+other_size]
        np.savez(output_path+"/validation.npz", images=[i[0] for i in validation], labels=[int(i[1]) for i in validation])
        print('Successfully written validation.npz')

        testing = data[training_size+other_size:len(data)]
        np.savez(output_path+"/testing.npz", images=[i[0] for i in testing], labels=[int(i[1]) for i in testing])
        print('Successfully written testing.npz')


# dataset class
class MosaicDataset(Dataset):
    def __init__(self, path_to_npz, ds_type):
        print(f"Preparing to load {ds_type} dataset from \'{path_to_npz}\'")
        self.type = ds_type
        arrays = np.load(path_to_npz)

        self.data = arrays['images']
        self.labels = torch.from_numpy(arrays['labels'])

        print(f"There are {torch.sum(self.labels)} of class 1 and {self.__len__()-torch.sum(self.labels)} of class 0")

        # go from HxWxC to normalized (0-1) CxHxW for input to ResNet
        self.transform = ToTensor()

        # Calculate mean and std for each channel
        print("Computing means and stds for each rgb channel...")
        rgb_mean = [self.data[:,:,:,0].mean()/255, self.data[:,:,:,1].mean()/255, self.data[:,:,:,2].mean()/255]
        rgb_mean = [i.item() for i in rgb_mean] # convert from 1x1 np array to floats

        rgb_std = [self.data[:,:,:,0].std()/255, self.data[:,:,:,1].std()/255, self.data[:,:,:,2].std()/255]
        rgb_std = [i.item() for i in rgb_std] # convert from 1x1 np array to floats
        
        self.normalize = Normalize(mean=rgb_mean, std=rgb_std)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = self.normalize(self.transform(self.data[index]))
        label = self.labels[index]

        return [image, label]


if __name__ == '__main__':
    #save_from_index('/home/joao/dev/MLAudio/shotspotter/data/mosaic_index.csv', '/home/joao/dev/MLAudio/shotspotter/data/dataset')

    ds = MosaicDataset('data/dataset/training.npz', 'training')
    print(ds[0])