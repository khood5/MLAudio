import argparse
import speech2spikes
import torchaudio
import torch
import random
import numpy as np
import os
import pywt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--dataset_cap', required=True)
parser.add_argument('-p', '--save_path', required=True)
parser.add_argument('-m', '--mode', choices=['s2s', 'samples', 'dwt', 'spec'], required=True)

args = parser.parse_args()

PATH_GUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/gunshots'
PATH_NOGUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/genBackgrounds'

s2s = speech2spikes.S2S()
s2s._default_spec_kwargs = {
    "sample_rate": 12000,
    "n_mels": 20,
    "n_fft": 512,
    "f_min": 20,
    "f_max": 4000,
    "hop_length": 10,
}
s2s.transform = torchaudio.transforms.MelSpectrogram(**s2s._default_spec_kwargs)

DATASET_CAP = int(args.dataset_cap)
DWT_LEVELS = 7
DWT_TIMESTEPS = 900

SPEC_FREQ_BIN_COUNT = 25

gunshot_file_paths = [PATH_GUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_GUNSHOT_SOUNDS)][:DATASET_CAP//2]
print(f'We have {len(gunshot_file_paths)} gunshot audio files')
nogunshot_file_paths = [PATH_NOGUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_NOGUNSHOT_SOUNDS)][:DATASET_CAP//2]
print(f'We have {len(nogunshot_file_paths)} background only audio files')

def to_spikes(paths_list, labels, mode='s2s'):
    print("Reading audio files...")
    data = []

    if mode == 's2s':
        for p in paths_list:
            samples, rate = torchaudio.load(p)
            data.append(samples)

        print("Running speech2spikes...")
        # Note: looks like it works when audio is both single and dual channel regardless.
        # Also, why is some of the audio dual channel whereas some is single channel - investigate
        trains, targets = s2s([(data[i], torch.tensor(labels[i])) for i in range(len(paths_list))])

        print(trains.shape)

        all_spikes = []

        for train in trains:
            spikes = []
            spike_id_count = 0
            for i in range(2):
                for channel in train[i]:
                    pos_channel, neg_channel = [], []
                    pos_id, neg_id = spike_id_count, spike_id_count+1
                    for t in range(len(channel)):
                        if channel[t] == 1:
                            pos_channel.append((pos_id, t, 1))
                            neg_channel.append((neg_id, t, 0))
                        elif channel[t] == -1:
                            neg_channel.append((neg_id, t, 1))
                            pos_channel.append((pos_id, t, 0))
                        else:
                            pos_channel.append((pos_id, t, 0))
                            neg_channel.append((neg_id, t, 0))
        
                    spikes.append(pos_channel.copy())
                    spikes.append(neg_channel.copy())

                    spike_id_count += 2

            all_spikes.append(spikes.copy())

    elif mode == 'samples':
        for p in paths_list:
            samples, rate = torchaudio.load(p, normalize=False)

            if samples.shape[0] == 2: # if dual channel, only take left
                samples = samples[0, :]
            else:
                samples = samples[0]

            # for some reason some have 23999 samples instead of exactly 24000
            if(len(samples) < 24000):
                samples = torch.cat((samples, torch.tensor([0])))

            data.append(samples)

        data = np.array(data, dtype=np.float64)
        data_max = data.max()
        data_min = data.min()

        norm_data = (data-data_min) / (data_max-data_min) # normalize 0-1
        
        # Just to match what I had before, put this in all_spikes list of arrays
        all_spikes = norm_data.tolist()
        targets = np.array(labels)

    elif mode == 'dwt':
        all_spikes = []
        targets = np.array(labels)
        for p in paths_list:
            samples, rate = torchaudio.load(p, normalize=False)

            # same procedure as 'samples' mode
            if samples.shape[0] == 2: samples = samples[0, :]
            else: samples = samples[0]
            if(len(samples) < 24000):
                samples = torch.cat((samples, torch.tensor([0])))

            coeffs = pywt.wavedec(samples, 'db1', level=DWT_LEVELS)

            # NOTE 1: that this will leave out the approximation coefficient, it carries information about the lowest
            # frequency band. I did it the same way for the mosaic CNN version, so maybe we just don't need it
            # NOTE 2: we are only taking the magnitude (hence np.abs) of the coefficients, same as for the mosaic CNN
            # NOTE 3: for my future reference, the code below just repeats the coefficient values at level X so that they all
            # align and match the D1 coeffs' size and timestep  
            accum = np.abs(np.array([coeffs[-1]]))
            for i in range(DWT_LEVELS - 1):
                current_coef = coeffs[DWT_LEVELS - 1 - i]
                r = np.abs(np.array([np.repeat(current_coef, pow(2, i + 1))]))
                r = r[:, 0:rate]
                accum = np.concatenate([accum, r])
            
            # min max normalize
            global_min = accum.min() 
            global_max = accum.max() 

            accum = (accum-global_min) / (global_max - global_min)

            timestep_skip = rate//DWT_TIMESTEPS

            # note that the timestep count will not be exact, timestep skip just approximates to the closest we can get 
            # without missing information at the end
            channels = [[] for i in range(DWT_LEVELS)]
            for i in range(rate//timestep_skip + 1):
                for j in range(DWT_LEVELS):
                    channels[j].append(accum[j][i*timestep_skip])

            channels = np.array(channels)
            all_spikes.append(channels)

            #np.savez('./test450timesteps.npz', channels=channels)
    
    elif mode == 'spec':
        all_spikes = []
        targets = np.array(labels)

        for p in paths_list:
            samples, rate = torchaudio.load(p, normalize=False)

            # same procedure as 'samples' mode
            if samples.shape[0] == 2: samples = samples[0, :]
            else: samples = samples[0]
            if(len(samples) < 24000):
                samples = torch.cat((samples, torch.tensor([0])))

            # freq bin count is nfft//2 + 1
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=(2*SPEC_FREQ_BIN_COUNT-2))

            samples = samples.to(torch.float64)
            spec = spec_transform(samples)

            # min max normalize
            global_min = spec.min() 
            global_max = spec.max() 

            spec = (spec-global_min) / (global_max - global_min)

            all_spikes.append(spec)

    return all_spikes, targets

# make sure path is .npz for consistency
def write_spikes_to_disk(path, train, train_labels, val, val_labels, test, test_labels):
    np.savez(path, train_set=train, validation_set=val, test_set=test, train_labels=train_labels,
            validation_labels=val_labels, test_labels=test_labels)

# generate spikes
p1 = [(i, 1) for i in gunshot_file_paths]
p2 = [(i, 0) for i in nogunshot_file_paths]

pairs = p1+p2 # path to sound - label tuples
random.shuffle(pairs)

spikes, labels = to_spikes([i[0] for i in pairs], [i[1] for i in pairs], mode=args.mode)

# training/validation/test split
train_cutoff_index = int(DATASET_CAP*0.8)
test_val_cutoff_offset = int(DATASET_CAP*0.1)
training_spikes = spikes[0:train_cutoff_index]
training_labels = labels[0:train_cutoff_index]

validation_spikes = spikes[train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]
validation_labels = labels[train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]

test_spikes = spikes[train_cutoff_index+test_val_cutoff_offset:]
test_labels = labels[train_cutoff_index+test_val_cutoff_offset:]

print("Writing to disk...")
write_spikes_to_disk(args.save_path, training_spikes, training_labels, validation_spikes, validation_labels, test_spikes, test_labels)