from PIL import Image
import numpy as np
import random

# expects index csv where column 1 is file path and column 2 is class
def load_from_index(index_path, output_path):
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
        np.savez(output_path+"/training.npz", training=[i[0] for i in training], labels=[i[1] for i in training])
        print('Successfully written training.npz')

        validation = data[training_size:training_size+other_size]
        np.savez(output_path+"/validation.npz", validation=[i[0] for i in validation], labels=[i[1] for i in validation])
        print('Successfully written validation.npz')

        testing = data[training_size+other_size:len(data)]
        np.savez(output_path+"/testing.npz", testing=[i[0] for i in testing], labels=[i[1] for i in testing])
        print('Successfully written testing.npz')


if __name__ == '__main__':
    load_from_index('/home/joao/dev/MLAudio/shotspotter/data/mosaic_index.csv', '/home/joao/dev/MLAudio/shotspotter/data/dataset')