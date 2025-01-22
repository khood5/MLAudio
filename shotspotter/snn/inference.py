import neuro
import risp
import argparse
import json
from common import risp_config, read_spikes_from_disk
import time

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--dataset', required=True)
parser.add_argument('-n', '--network', required=True)
args = parser.parse_args()

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


def compute_fitness(spikes, labels, display_per_class=False, reconstruct_spikes=False):
    # See train_script
    if reconstruct_spikes:
        rec_spikes = []

        for i in range(len(spikes)):
            rec_spikes.append([])
            
            for j in range(len(spikes[i])):
                rec_spikes[i].append([])
                
                for spk in spikes[i][j]:
                    if spk[2] == 1:
                        rec_spikes[i][j].append(neuro.Spike(spk[0], spk[1], spk[2]))

        spikes = rec_spikes
    
    correct = 0
    per_class = {0: 0, 1: 0}
    for i in range(len(spikes)):
        proc.clear_activity()

        for c in spikes[i]: # spikes[i] is a single training sample
            proc.apply_spikes(c)

        proc.run(1000)

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

print(f'Accuracy on the test set is {test_fit/len(test_labels)}')
print(f'Computing for all {len(test_labels)} samples in test set took {time.time()-t0:.2f} seconds')