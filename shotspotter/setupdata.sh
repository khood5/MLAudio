#!/bin/bash

# change sample size and paths to match your setup

mkdir data/mosaic/gunshot
mkdir data/mosaic/nogunshot
mkdir data/dataset
mkdir data/gunshots
mkdir data/genBackgrounds

python make_index.py -d /home/joao/dev/data/shotspotter/sortedBackground -o data/raw_backgrounds_index.py
python make_index.py -d /home/joao/dev/data/shotspotter/sortedGunshot -o data/raw_gunshots_index.py

python ../data/makeBackgroundAudio.py -l 2 -sr 12000 data/raw_backgrounds_index.py 10000 data/genBackgrounds
python make_index.py -d data/genBackgrounds/ -o data/backgrounds_index.csv

python ../data/makeGunshotAudio.py -n 10000 -sr 12000 data/backgrounds_index.csv data/raw_gunshots_index.py data/gunshots
python make_index.py -d data/gunshots/ -o data/gunshot_index.csv

python make_mosaic.py -s data/gunshots/ -l 1 -o data/mosaic/gunshot -i data/mosaic_index.csv
python make_mosaic.py -s data/genBackgrounds/ -l 0 -o data/mosaic/nogunshot -i data/mosaic_index.csv