from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor

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


# TODO: FIX HERE, we can't load and apply transform all at once, do it in getitem per item

# dataset class
class MosaicDataset(Dataset):
    def __init__(self, path_to_npz, ds_type):
        self.type = ds_type
        print(f"Loading {self.type} dataset from \'{path_to_npz}\'")
        arrays = np.load(path_to_npz)

        self.data = arrays['images']
        self.labels = torch.from_numpy(arrays['labels'])

        # go from HxWxC to normalized (0-1) CxHxW for input to ResNet
        self.data = [ToTensor()(d) for d in self.data]
        self.data = torch.stack(self.data)

        if torch.cuda.is_available():
            print('Moving tensors to GPU')
            self.data = self.data.to('cuda')
            self.labels = self.labels.to('cuda')
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return [self.data[index], self.labels[index]]


if __name__ == '__main__':
    save_from_index('/home/joao/dev/MLAudio/shotspotter/data/mosaic_index.csv', '/home/joao/dev/MLAudio/shotspotter/data/dataset')

    #ds = MosaicDataset('data/dataset/training.npz', 'training')
    #print(ds[0])