import numpy as np
import pandas as pd
from pydub import AudioSegment
import random
import argparse


def makeBackgroundAudio(audioBits):
    BackgroundAudio = AudioSegment.empty()
    samples = random.sample(audioBits, 1200) # shortest clip is 0.05secs so a sample of 1200 guarantees at least 60secs of audio
    while BackgroundAudio.duration_seconds < 60:
        path = samples[0][0]
        randomBit = AudioSegment.from_file(path, format="wav", codec="pcm_s32le")
        BackgroundAudio = BackgroundAudio + randomBit
        samples = samples[1:]
    BackgroundAudio = BackgroundAudio[:60 * 1000] # convert milliseconds to seconds
    return BackgroundAudio

def getListOfFiles(urbanIndex):
    paths = list(urbanIndex['Filename'])
    durations = list(urbanIndex['duration_secs'])
    folds = list(urbanIndex['fold'])
    backgoundBits = [(paths[i], durations[i]) for i in range(0, len(paths))]
    return backgoundBits

def main():
    parser = argparse.ArgumentParser(description='Generate background audio from the UrbanSound8k dataset.')
    
    # Add command-line arguments
    parser.add_argument('indexfile_path', type=str, help='Path to the index file')
    parser.add_argument('number_of_samples_to_make', type=int, help='Number of samples to generate')
    parser.add_argument('length', type=int, help='length of each sample')
    parser.add_argument('output_directory', type=str, help='Path to the output directory')
    args = parser.parse_args()

    indexfile_path = args.indexfile_path
    number_of_samples_to_make = args.number_of_samples_to_make
    length = args.length
    output_directory = args.output_directory


    


if __name__ == "__main__":
    main()