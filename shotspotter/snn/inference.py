import neuro
import risp
import argparse
import json
from common import risp_config, read_spikes_from_disk
import time
from power import watt_now
import threading
from make_dataset import to_spikes
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--dataset', required=True, help='if dataset path ends in .wav, then read in a single sample for inference (debugging)')
parser.add_argument('-n', '--network', required=True)
parser.add_argument('-m', '--mode', required=True, choices=['s2s', 'samples', 'dwt'])
parser.add_argument('--out_path', required=True)
parser.add_argument('-l', '--label', required=False, help='label for single sample, required if -p/--dataset is a single sample')
parser.add_argument('-t', '--threshold', required=True, help='threshold value to identify gunshot positive samples, should have been computed using validation set.')
parser.add_argument('-s', '--timesteps', required=True)
args = parser.parse_args()

MODE = args.mode
THRESHOLD = int(args.threshold)
PROC_TIMESTEPS = int(args.timesteps)

proc = risp.Processor(risp_config)

network = neuro.Network()
with open(args.network, 'r') as f:
    net_json = json.load(f)
network.from_json(net_json)

proc.load_network(network)
for out_id in network.as_json()['Outputs']:
    proc.track_neuron_events(out_id)

# load our spikes and labels
if '.wav' not in args.dataset:
    _, _, _, _, _, _, _, test_spikes, test_labels = read_spikes_from_disk(args.dataset)
else:
    if args.label is None:
        print('Label is required for single sample')
        exit()

    print('Converting single sample into spikes...')
    test_spikes, test_labels, _ = to_spikes([args.dataset], [int(args.label)], mode=MODE, need_time_data=False)
    test_spikes = np.array(test_spikes)

# thread stuff
stop = threading.Event()
def record_consumption():
    while not stop.is_set():
        #with open(args.out_path, 'a') as f:
            #f.write(f'{watt_now()}\n')

        time.sleep(0.005)

thr = threading.Thread(target=record_consumption)
thr.start()

def compute_fitness(spikes, labels, reconstruct_spikes=False):
    if reconstruct_spikes:
        rec_spikes = []

        if MODE == 's2s':
            for i in range(len(spikes)):
                rec_spikes.append([])

                for j in range(len(spikes[i])):
                    rec_spikes[i].append([])
                    
                    for spk in spikes[i][j]:
                        if spk[2] == 1:
                            rec_spikes[i][j].append(neuro.Spike(spk[0], spk[1], spk[2]))
                    
        elif MODE == 'samples':
            for i in range(len(spikes)): 
                rec_spikes.append([])
                
                for j in range(len(spikes[0])):
                    # id, time, value
                    rec_spikes[i].append(neuro.Spike(0, j, spikes[i][j]))

        elif MODE == 'dwt':
            for i in range(spikes.shape[0]):
                rec_spikes.append([])

                for j in range(spikes.shape[1]):
                    rec_spikes[i].append([])

                    for k in range(spikes.shape[2]):
                        #print(f'Creating Spike({j}, {k}, {shared_spikes_arr[i][j][k]})')
                        rec_spikes[i][j].append(neuro.Spike(j, k, spikes[i][j][k]))

        spikes = rec_spikes

    timesteps_from_data = len(spikes[0][0]) if MODE != 'samples' else 24000
    
    # NOTE: has new meaning here (compared to train_script.py): just the total number of gunshot output neuron spikes regardless of label
    gunshot_spikes = [] # each entry is tuple where (label, gunshot_spikes) 
    for i in range(len(spikes)):
        proc.clear_activity()

        # apparently clear_activity resets this
        proc.track_output_events(0)
        proc.track_output_events(1)

        if MODE == 's2s' or MODE == 'dwt' or MODE == 'spec':
            for c in spikes[i]: # spikes[i] is a single training sample
                proc.apply_spikes(c)
        elif MODE == 'samples':
            proc.apply_spikes(spikes[i])

        proc.run(PROC_TIMESTEPS)
        gunshot_spikes.append((int(labels[i]), proc.output_counts()[1]))

    return np.array(gunshot_spikes)

t0 = time.time()

test_fit = compute_fitness(test_spikes, test_labels, reconstruct_spikes=True)

stop.set()
thr.join()

correct = 0
fn, fp = 0, 0
total = len(test_fit)

for i in range(total):
    if test_fit[i][0] == 1 and test_fit[i][1] > THRESHOLD:
        correct += 1
    elif test_fit[i][0] == 1 and test_fit[i][1] <= THRESHOLD:
        fn += 1
    elif test_fit[i][0] == 0 and test_fit[i][1] <= THRESHOLD:
        correct += 1
    elif test_fit[i][0] == 0 and test_fit[i][1] > THRESHOLD:
        fp += 1

print(f'Test set (size={total}) results\n')
print(f'Inference took {time.time()-t0:.2f} seconds')


print(f'{correct/total:.3f} accuracy {correct} out of {total}')
print(f'False positive count: {fp}')
print(f'False negative count: {fn}')