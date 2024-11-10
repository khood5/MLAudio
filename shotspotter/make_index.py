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
parser.add_argument('-o', '--output', default=OUTPUT_DIR+'index.csv', help='file path + name of output csv')
args = parser.parse_args()

if args.directory[-1] != '/':
    args.directory += '/'

with open(args.output, 'w') as f:
    full_paths = [args.directory+filename+"\n" for filename in os.listdir(args.directory)]
    f.writelines(full_paths)
