import argparse
import speech2spikes
import torchaudio
import torch
import random
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--dataset_cap', required=True)
parser.add_argument('-p', '--save_path', required=True)

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
    "hop_length": 80,
}
s2s.transform = torchaudio.transforms.MelSpectrogram(**s2s._default_spec_kwargs)

DATASET_CAP = int(args.dataset_cap)

gunshot_file_paths = [PATH_GUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_GUNSHOT_SOUNDS)][:DATASET_CAP//2]
print(f'We have {len(gunshot_file_paths)} gunshot audio files')
nogunshot_file_paths = [PATH_NOGUNSHOT_SOUNDS+'/'+fn for fn in os.listdir(PATH_NOGUNSHOT_SOUNDS)][:DATASET_CAP//2]
print(f'We have {len(nogunshot_file_paths)} background only audio files')

def to_spikes(paths_list, labels):
    print("Reading audio files...")
    data = []
    for p in paths_list:
        samples, rate = torchaudio.load(p)
        data.append(samples)

    print("Running speech2spikes...")
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

spikes, labels = to_spikes([i[0] for i in pairs], [i[1] for i in pairs])

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