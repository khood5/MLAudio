import neuro
import risp
import argparse
import json
from common import risp_config, read_spikes_from_disk
import time
from power import watt_now
import threading

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--dataset', required=True)
parser.add_argument('-n', '--network', required=True)
parser.add_argument('-m', '--mode', required=True, choices=['s2s', 'samples', 'dwt'])
parser.add_argument('--out_path', required=True)
args = parser.parse_args()

MODE = args.mode

proc = risp.Processor(risp_config)

network = neuro.Network()
with open(args.network, 'r') as f:
    net_json = json.load(f)
network.from_json(net_json)

proc.load_network(network)
for out_id in network.as_json()['Outputs']:
    proc.track_neuron_events(out_id)

# load our spikes and labels
_, _, _, _, test_spikes, test_labels = read_spikes_from_disk(args.dataset)

# thread stuff
stop = threading.Event()
def record_consumption():
    while not stop.is_set():
        #with open(args.out_path, 'a') as f:
            #f.write(f'{watt_now()}\n')

        time.sleep(0.005)

thr = threading.Thread(target=record_consumption)
#thr.start()

def compute_fitness(spikes, labels, display_per_class=False, reconstruct_spikes=False):
    # See train_script
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
            # NOTE: inference for this mode is untested

            # for this one our dimensions are: (data sample #, 24k audio samples)
            for i in range(len(spikes)): 
                rec_spikes.append([])
                
                for j in range(len(spikes[0])):
                    # id, time, value
                    rec_spikes[i].append(neuro.Spike(0, j, spikes[i][j]))

        elif MODE == 'dwt':
            # here dimensions will be (7, timesteps) per sample
            for i in range(spikes.shape[0]):
                rec_spikes.append([])

                for j in range(spikes.shape[1]):
                    rec_spikes[i].append([])

                    for k in range(spikes.shape[2]):
                        #print(f'Creating Spike({j}, {k}, {shared_spikes_arr[i][j][k]})')
                        rec_spikes[i][j].append(neuro.Spike(j, k, spikes[i][j][k]))

        spikes = rec_spikes
    
    correct = 0
    per_class = {0: 0, 1: 0}
    for i in range(len(spikes)):
        proc.clear_activity()

        for c in spikes[i]: # spikes[i] is a single training sample
            proc.apply_spikes(c)

        proc.run(20)

        out_counts = proc.output_counts()

        prediction = 0 if out_counts[0] > out_counts[1] else 1

        if prediction == labels[i]:
            correct += 1
            per_class[prediction] += 1

    if display_per_class:
        print(f'For class 0: {per_class[0]}/{(labels == 0).sum()}')
        print(f'For class 1: {per_class[1]}/{(labels == 1).sum()}')

    return correct

t0 = time.time()

test_fit = compute_fitness(test_spikes, test_labels, display_per_class=True, reconstruct_spikes=True)

stop.set()
thr.join()

print(f'Accuracy on the test set is {test_fit/len(test_labels)}')
print(f'Computing for all {len(test_labels)} samples in test set took {time.time()-t0:.2f} seconds')