from pydub import AudioSegment
import random
import argparse
import csv
from tqdm import tqdm
import uuid
from multiprocessing import Process, Manager
import argparse
import sys

def makeGunshotAudio(sampleGunshotAudio, backgroundAudio, outputPath, sharedRows, index):
    BackgroundAudio = AudioSegment.from_file(backgroundAudio[index], format="wav")
    gunshotPath = random.sample(sampleGunshotAudio, 1)[0]
    gunshot = AudioSegment.from_file(gunshotPath, format="wav")

    gunshotInjectionPoint = random.uniform(1, BackgroundAudio.duration_seconds - gunshot.duration_seconds)
    GunshotAudio = BackgroundAudio.overlay(gunshot, position=gunshotInjectionPoint*1000) # times 1000 to convert from mili to secs

    GunshotAudio.export(outputPath, format="wav", codec="pcm_s32le").close()
    sharedRows[index] = [str(outputPath), str(gunshotInjectionPoint), backgroundAudio[index], gunshotPath]

def main():
    # Add command-line arguments
    parser = argparse.ArgumentParser(description='Generate gunshot and background audio from list of background audio files (of desired length) and list of gunshots.')

    parser.add_argument('background_index', type=str, help='Path to the index file (csv file with list of paths to background sound files)')
    parser.add_argument('gunshot_index', type=str, help='Path to the index file (csv file with list of paths to gunshot sound files)')
    parser.add_argument('output_directory', type=str, help='Path to the output directory')
    parser.add_argument('-n', '--number_of_samples_to_make', default=-1, type=int, help='Number of samples to generate (max equal to the number of background files) default= all files in background_index')
    parser.add_argument('-p', '--num_processes', nargs='?', default=2, type=int, help='number of processes to use default=2')
    args = parser.parse_args()

    gunshotSamples = []
    with open(args.gunshot_index, newline='') as csvfile:
        gunshotIndexFilePaths = csv.reader(csvfile, delimiter=',')
        for row in gunshotIndexFilePaths:
            gunshotSamples.append(row[0])

    backgroundSounds = []
    with open(args.background_index, newline='') as csvfile:
        backgroundIndexFilePaths = csv.reader(csvfile, delimiter=',')
        for row in backgroundIndexFilePaths:
            backgroundSounds.append(row[0])
    number_of_samples_to_make = args.number_of_samples_to_make if args.number_of_samples_to_make != -1 else len(backgroundSounds)
    if(number_of_samples_to_make > len(backgroundSounds)):
        print("ERROR: number_of_samples_to_make must be euqal or smaller then the number of background audio files in the backgroundIndex")
        print(f"Requested \033[1m {args.number_of_samples_to_make} \033[0m number_of_samples_to_make")
        print(f"But only found \033[1m {len(backgroundSounds)} \033[0m background audio files in background_index")
        sys.exit(1)  # abort because of error

    remainingSamples = args.number_of_samples_to_make
    index = 0
    manager = Manager()
    gunshotWithBackgroundIndexRows = manager.list([[]]*args.number_of_samples_to_make) 
    with tqdm(total=args.number_of_samples_to_make) as pbar:
        remainingSamples = args.number_of_samples_to_make
        while remainingSamples > 0:
            processList = []
            numProcesses = min(args.num_processes, remainingSamples)
            for _ in range(numProcesses):
                numProcesses = min(args.num_processes, remainingSamples)
                process = Process(target=makeGunshotAudio, args=(gunshotSamples, 
                                                                 backgroundSounds, 
                                                                 f"{args.output_directory}/gunshot_{uuid.uuid4()}.wav",
                                                                 gunshotWithBackgroundIndexRows, 
                                                                 index
                                                                 )
                )
                processList.append(process)
                process.start()
                index = index + 1
            for process in processList:
                process.join()
            pbar.update(numProcesses)
            remainingSamples = remainingSamples - numProcesses
            
    file = open('gunshotIndex.csv', 'w+', newline ='')
    with file:    
        write = csv.writer(file)
        write.writerow(["File", "Gunshot Timestamp", "background source file", "gunshot source file"]) # header
        write.writerows([i for i in gunshotWithBackgroundIndexRows]) 

if __name__ == "__main__":
    main()