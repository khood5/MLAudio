import argparse
import eons
import neuro
import risp
import speech2spikes
import os
import torchaudio
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import functools
import json

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--num_processes', required=True)

args = parser.parse_args()

# Constants and configs ----------------------------------------------------------------------------
PATH_GUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/gunshots'
PATH_NOGUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/genBackgrounds'

s2s = speech2spikes.S2S()
s2s._default_spec_kwargs = {
    "sample_rate": 12000,
    "n_mels": 20,
    "n_fft": 512,
    "f_min": 20,
    "f_max": 4000,
    "hop_length": 80,
}
s2s.transform = torchaudio.transforms.MelSpectrogram(**s2s._default_spec_kwargs)

# Some constants
NUM_INPUT_NEURONS = 80 # see paper
NUM_OUTPUT_NEURONS = 2
NUM_SYNAPSES = 1000
NUM_HIDDEN_NEURONS = 250
POP_SIZE = 70

MOA = neuro.MOA()
MOA.seed(23456789, '')

DATASET_CAP = 1200
NUM_PROCESSES = args.num_processes