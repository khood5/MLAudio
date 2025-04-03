import argparse
import speech2spikes
import torchaudio
import torch
import random
import numpy as np
import os
import pywt
import noisereduce


# stuff required for to_spikes
DWT_LEVELS = 7
DWT_TIMESTEPS = 450

SPEC_FREQ_BIN_COUNT = 25

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

# will return (-1, -1) for background files (they have no gunshot injection data)
def get_time_data(filename):
    if filename.lower().find('background') != -1:
        return (-1, -1)
    return raw_time_data[filename.split('/')[-1]]

def to_spikes(paths_list, labels, mode='s2s', need_time_data=True):
    print("Reading audio files...")
    data = []

    print(f"Running '{mode}' encoding...")
    gunshot_time_data = []
    if mode == 's2s':
        for p in paths_list:
            if need_time_data:
                gunshot_time_data.append(get_time_data(p))

            samples, rate = torchaudio.load(p)
            data.append(samples)

        # Note: looks like it works when audio is both single and dual channel regardless.
        trains, targets = s2s([(data[i], torch.tensor(labels[i])) for i in range(len(paths_list))])

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
            if need_time_data:
                gunshot_time_data.append(get_time_data(p))

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
            if need_time_data:
                gunshot_time_data.append(get_time_data(p))

            samples, rate = torchaudio.load(p, normalize=True)

            # same procedure as 'samples' mode
            if samples.shape[0] == 2: samples = samples[0, :]
            else: samples = samples[0]
            if(len(samples) < 24000):
                samples = torch.cat((samples, torch.tensor([0])))

            samples = noisereduce.reduce_noise(y=samples, sr=rate) # testing this because I had it on in the ResNet version dataset

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
            
            timestep_skip = rate//DWT_TIMESTEPS

            # note that the timestep count will not be exact, timestep skip just approximates to the closest we can get 
            # without missing information at the end
            channels = [[] for i in range(DWT_LEVELS)]
            for i in range(rate//timestep_skip):
                for j in range(DWT_LEVELS):
                    channels[j].append(accum[j][i*timestep_skip])

            channels = np.array(channels)

            channels = (channels - channels.min()) / (channels.max()-channels.min())

            all_spikes.append(channels)
        
        # now we find global max and min and normalize using that
        # global_min = 0
        # global_max = 0
        # for s in all_spikes:
        #     if s.min() < global_min: global_min = s.min()
        #     if s.max() > global_max: global_max = s.max()

        # # now normalize everything
        # for i in range(len(all_spikes)):
        #     all_spikes[i] = (all_spikes[i]-global_min) / (global_max - global_min)
        
    elif mode == 'spec':
        all_spikes = []
        targets = np.array(labels)

        for p in paths_list:
            if need_time_data:
                gunshot_time_data.append(get_time_data(p))

            samples, rate = torchaudio.load(p, normalize=True)

            # same procedure as 'samples' mode
            if samples.shape[0] == 2: samples = samples[0, :]
            else: samples = samples[0]
            if(len(samples) < 24000):
                samples = torch.cat((samples, torch.tensor([0])))

            samples = torch.tensor(noisereduce.reduce_noise(y=samples, sr=rate)) # testing this because I had it on in the ResNet version dataset

            # freq bin count is nfft//2 + 1
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=(2*SPEC_FREQ_BIN_COUNT-2))

            samples = samples.to(torch.float64)
            spec = spec_transform(samples)

            # per sample normalize
            
            #spec = torch.log10(spec+1e-6) # log to compress scale
            
            #spec = (spec - spec.min()) / (spec.max() - spec.min())

            all_spikes.append(spec)

        #GLOBAL NORMALIZATION
        global_min = 0
        global_max = 0
        for s in all_spikes:
            if s.min() < global_min: global_min = s.min()
            if s.max() > global_max: global_max = s.max()

        # now normalize everything
        for i in range(len(all_spikes)):
            all_spikes[i] = (all_spikes[i]-global_min) / (global_max - global_min)

    return all_spikes, targets, gunshot_time_data

# make sure path is .npz for consistency
def write_spikes_to_disk(path, train, train_labels, train_gunshot_data, val, val_labels, val_gunshot_data,
val_filenames, test, test_labels):
    np.savez(path, train_set=train, validation_set=val, test_set=test, train_labels=train_labels,
            validation_labels=val_labels, test_labels=test_labels, train_gunshot_data=train_gunshot_data,
            validation_gunshot_data=val_gunshot_data, validation_filenames=val_filenames)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--dataset_cap', required=True)
    parser.add_argument('-p', '--save_path', required=True)
    parser.add_argument('-m', '--mode', choices=['s2s', 'samples', 'dwt', 'spec'], required=True)

    args = parser.parse_args()

    PATH_GUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/gunshotsNew'
    # we need this next one to get the timestamps for gunshot injection and duration
    # note that this is the file that is output from /data/makeGunshotAudio.py
    PATH_GUNSHOT_INDEX = '/home/joao/dev/MLAudio/shotspotter/data/gunshotsNewIndex.csv'
    PATH_NOGUNSHOT_SOUNDS = '/home/joao/dev/MLAudio/shotspotter/data/genBackgrounds'

    DATASET_CAP = int(args.dataset_cap)

    gunshot_file_paths = [PATH_GUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_GUNSHOT_SOUNDS)][:DATASET_CAP//2]
    print(f'We have {len(gunshot_file_paths)} gunshot audio files')
    nogunshot_file_paths = [PATH_NOGUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_NOGUNSHOT_SOUNDS)][:DATASET_CAP//2]
    print(f'We have {len(nogunshot_file_paths)} background only audio files')

    # get gunshot injection time data 
    raw_time_data = {} # we'll have key=filename, value=(injection_time, duration_time)
    with open(PATH_GUNSHOT_INDEX, 'r') as f:
        lines = [i.replace('\n', '') for i in f.readlines()]
        lines = [i.split(',') for i in lines]
        raw_time_data = {l[0].split('/')[-1]:(float(l[1]), float(l[2])) for l in lines[1:]}

    # generate spikes
    p1 = [(i, 1) for i in gunshot_file_paths]
    p2 = [(i, 0) for i in nogunshot_file_paths]

    pairs = p1+p2 # path to sound - label tuples
    random.shuffle(pairs)

    spikes, labels, gunshot_data = to_spikes([i[0] for i in pairs], [i[1] for i in pairs], mode=args.mode)

    # training/validation/test split
    train_cutoff_index = int(DATASET_CAP*0.8)
    test_val_cutoff_offset = int(DATASET_CAP*0.1)
    training_spikes = spikes[0:train_cutoff_index]
    training_labels = labels[0:train_cutoff_index]
    training_gunshot_data = gunshot_data[0:train_cutoff_index]

    validation_spikes = spikes[train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]
    validation_labels = labels[train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]
    validation_gunshot_data = gunshot_data[train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]
    validation_filenames = [i[0] for i in pairs][train_cutoff_index:train_cutoff_index+test_val_cutoff_offset]

    test_spikes = spikes[train_cutoff_index+test_val_cutoff_offset:]
    test_labels = labels[train_cutoff_index+test_val_cutoff_offset:]

    print("Writing to disk...")
    write_spikes_to_disk(args.save_path, training_spikes, training_labels, training_gunshot_data,
    validation_spikes, validation_labels, validation_gunshot_data, validation_filenames, test_spikes, test_labels)