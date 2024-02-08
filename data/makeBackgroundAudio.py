import numpy as np
import pandas as pd
from pydub import AudioSegment
import random
import argparse
import csv
from tqdm import tqdm
import threading
import uuid

def makeBackgroundAudio(audioBits, outputPath, length=60):
    BackgroundAudio = AudioSegment.empty()
    samples = random.sample(audioBits, 1200) # shortest clip is 0.05secs so a sample of 1200 guarantees at least 60secs of audio
    while BackgroundAudio.duration_seconds < length:
        path = samples[0]
        randomBit = AudioSegment.from_file(path, format="wav", codec="pcm_s32le")
        BackgroundAudio = BackgroundAudio + randomBit
        samples = samples[1:]
    BackgroundAudio = BackgroundAudio[:60 * 1000] # convert milliseconds to seconds
    BackgroundAudio.export(outputPath, format="wav", codec="pcm_s32le")

def main():
    parser = argparse.ArgumentParser(description='Generate background audio from the UrbanSound8k dataset.')
    
    # Add command-line arguments
    parser.add_argument('indexfile_path', type=str, help='Path to the index file (csv file with list of paths to background sound files)')
    parser.add_argument('number_of_samples_to_make', type=int, help='Number of samples to generate')
    parser.add_argument('-l', '--length', nargs='?', default=60, type=int, help='length of each sample default=60')
    parser.add_argument('-t', '--num_threads', nargs='?', default=2, type=int, help='number of threads to use default=2')
    parser.add_argument('output_directory', type=str, help='Path to the output directory')
    args = parser.parse_args()
    
    print(f"Reading sound bytes from {args.indexfile_path}")

    backgoundSoundBytes = []
    with open(args.indexfile_path, newline='') as csvfile:
        backgoundSoundBytesIndexFile = csv.reader(csvfile, delimiter=',')
        for row in backgoundSoundBytesIndexFile:
            backgoundSoundBytes.append(row[0])

    print(f"Making {args.length} sec long background samples")
    print(f"Making {args.number_of_samples_to_make} total smaples")
    print(f"Putting samples in {args.output_directory}")
    remainingSamples = args.number_of_samples_to_make
    outputFileNames = []
    with tqdm(total=args.number_of_samples_to_make) as pbar:
        while remainingSamples > 0: 
        # in tqdm(range(args.number_of_samples_to_make)):
            num_threads = args.num_threads if (remainingSamples - args.num_threads) > 0 else remainingSamples
            threads = []
            for i in range(num_threads):
                # use uuid.uuid4() to make a random UUID so file names dont collided 
                outputFileNames.append(f"{args.output_directory}/background_{uuid.uuid4()}.wav")
                thread = threading.Thread(target=makeBackgroundAudio, args=(backgoundSoundBytes, outputFileNames[-1], args.length,))
                threads.append(thread)
                thread.start()
            # makeBackgroundAudio(backgoundSoundBytes, args.length).export(f"{args.output_directory}/background{i}.wav", format="wav", codec="pcm_s32le")
            for thread in threads:
                thread.join()
            pbar.update(num_threads)
            remainingSamples = remainingSamples - num_threads
    file = open('backgroundIndex.csv', 'w+', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerows([[i] for i in outputFileNames]) # Strings must be placed in separate arrays to prevent each character from being treated as an individual column.


if __name__ == "__main__":
    main()