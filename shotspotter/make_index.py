# Make index CSV files for raw background, raw gunshot or combined background (made via makeBackgroundAudio.py) 
# sound files

import os 
import argparse

#BACKGROUND_SRC_PATH = os.path.expanduser('~')+'/data/background_src/'
#GUNSHOT_SRC_PATH = os.path.expanduser('~')+'/data/gunshot_src/'
OUTPUT_DIR = os.path.expanduser('~')+'/gunshot-resnet/data/'

parser = argparse.ArgumentParser(description='''generate the csv files used for input for makeBackgroundAudio.py or makeGunshotAudio.py
example usage: python make_index.py -d ~/data/gunshot_src/ -o ../data/raw_gunshot.csv''')
parser.add_argument('-d', '--directory', required=True, help='directory to index')
parser.add_argument('-o', '--output', required=False, default=OUTPUT_DIR+'index.csv', help='file path + name of output csv')
parser.add_argument('-m', '--mosaic_index', required=False, help='optional parameter to calculate append to mosaic index')
parser.add_argument('-l', '--label', required=False, help='optional parameter to specify labels to append with mosaic index')
args = parser.parse_args()

if args.directory[-1] != '/':
    args.directory += '/'

full_paths = [args.directory+filename+"\n" for filename in os.listdir(args.directory)]

if args.mosaic_index is not None:
    if args.label is None:
        print('Label is required to append/create mosaic index')
        exit()

    with open(args.mosaic_index, 'a') as f:
        for i in range(len(full_paths)):
            f.write(f'{full_paths[i].replace('\n', '')},{args.label}\n')

    exit()


with open(args.output, 'w') as f:
    f.writelines(full_paths)
