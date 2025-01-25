# script for picking a subset of size N from all of my generated audio samples
# I want to copy them somewhere else so we can generate datasets for the ResNets and SNN with the exact same audio samples to compare
from random import shuffle
import shutil
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', required=True, help='directory to copy our set to')
parser.add_argument('--gunshot_path', required=True)
parser.add_argument('--nogunshot_path', required=True)
parser.add_argument('--num_samples', required=True, help='must be even')
args = parser.parse_args()

DATASET_CAP = int(args.num_samples)
GUNSHOT_DIR_PATH = args.out_path+'/gunshot'
NOGUNSHOT_DIR_PATH = args.out_path+'/nogunshot'

os.mkdir(GUNSHOT_DIR_PATH)
os.mkdir(NOGUNSHOT_DIR_PATH)

gunshot_paths = [f'{args.gunshot_path}/{filename}' for filename in os.listdir(args.gunshot_path)]
nogunshot_paths = [f'{args.nogunshot_path}/{filename}' for filename in os.listdir(args.nogunshot_path)]
shuffle(gunshot_paths)
shuffle(nogunshot_paths)
gunshot_paths = gunshot_paths[:DATASET_CAP//2]
nogunshot_paths = nogunshot_paths[:DATASET_CAP//2]

# now let's copy from our synthetic dataset to the out path so we have our subset isolated
for i in range(DATASET_CAP//2):
    shutil.copy(gunshot_paths[i], f'{GUNSHOT_DIR_PATH}/{gunshot_paths[i].split('/')[-1]}')
    shutil.copy(nogunshot_paths[i], f'{NOGUNSHOT_DIR_PATH}/{nogunshot_paths[i].split('/')[-1]}')