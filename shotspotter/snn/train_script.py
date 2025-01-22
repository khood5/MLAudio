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
from common import risp_config, read_spikes_from_disk, network_details

# TODO: loading best network and injecting it into population
# TODO: also, look into how we are reading filenames to ensure randomness

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--num_processes', required=True)
parser.add_argument('-d', '--dataset_path', required=True)
parser.add_argument('-b', '--best_net_path', required=True) # json for bet network path
parser.add_argument('-l', '--log_path', required=True)
parser.add_argument('-e', '--epoch_count', required=True)
parser.add_argument('--synapse_count', required=True)
parser.add_argument('--hidden_count', required=True)
parser.add_argument('--num_mutations', required=True)
parser.add_argument('--random_factor', default='0.10', required=False)
parser.add_argument('--mutation_rate', default='0.9', required=False)

args = parser.parse_args()

# Constants and configs ----------------------------------------------------------------------------------------
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

NUM_INPUT_NEURONS = 80 # see paper
NUM_OUTPUT_NEURONS = 2
NUM_SYNAPSES = int(args.synapse_count)
NUM_HIDDEN_NEURONS = int(args.hidden_count)
POP_SIZE = 70

MOA = neuro.MOA()
MOA.seed(23456789, '')

NUM_PROCESSES = int(args.num_processes)

# Configure EONS
eons_param = {
    "starting_nodes": NUM_HIDDEN_NEURONS,
    "starting_edges": NUM_SYNAPSES,
    "merge_rate": 0,
    "population_size": POP_SIZE,
    "multi_edges": 0,
    "crossover_rate": 0.9,
    "mutation_rate": int(args.mutation_rate),
    "selection_type": "tournament",
    "tournament_size_factor": 0.1,
    "tournament_best_net_factor": 0.9,
    "random_factor": float(args.random_factor),
    "num_mutations": int(args.num_mutations),
    "node_mutations": { "Threshold": 1.0 },
    "net_mutations": { },
    "edge_mutations": { "Weight": 0.65 , "Delay": 0.35,  },
    "num_best" : 2,
    "add_node_rate": 0.75,
    "delete_node_rate": 0.25,
    "add_edge_rate": 0.75,
    "delete_edge_rate": 0.25,
    "node_params_rate": 2.5,
    "edge_params_rate": 2.5,
    "net_params_rate" : 0
}

proc = risp.Processor(risp_config)
eons_inst = eons.EONS(eons_param)

# Read data
training_spikes, training_labels, validation_spikes, validation_labels, test_spikes, test_labels = read_spikes_from_disk(args.dataset_path)

# set up template network  (inputs and outputs) for eons
template_net = neuro.Network()
template_net.set_properties(proc.get_network_properties())

for i in range(NUM_INPUT_NEURONS):
    node = template_net.add_node(i)
    node.set("Threshold", 1)
    template_net.add_input(i)

for i in range(NUM_INPUT_NEURONS, NUM_INPUT_NEURONS+NUM_OUTPUT_NEURONS):
    node = template_net.add_node(i)
    node.set("Threshold", 1)
    template_net.add_output(i)

proc.load_network(template_net)
# track neuron updates
for output_id in template_net.as_json()['Outputs']:
    proc.track_neuron_events(output_id)

# TRAINING -------------------------------------------------------------------------------------------

def compute_fitness(net, spikes, labels, display_per_class=False, reconstruct_spikes=False):
    # Explained in SPECIAL NOTE below, rebuild Spike instances to feed into processor
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
    
    proc.load_network(net)

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


EPOCH_COUNT = int(args.epoch_count)
eons_inst.set_template_network(template_net)
pop = eons_inst.generate_population(eons_param, MOA.random_integer())

best_fit_log = []
best_fit_validation_log = []
pop_fit_log = [] # mean
TRAINING_SET_SIZE = training_labels.shape[0]
VALIDATION_SET_SIZE = validation_labels.shape[0]
t0, t1 = 0, 0

# training loop
for i in range(EPOCH_COUNT):
    print(f'Starting epoch {i}...')
    t0 = time.time()

    compute_fitness_partial = functools.partial(compute_fitness, spikes=training_spikes, labels=training_labels, reconstruct_spikes=True)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        fits = p.map(compute_fitness_partial, 
                    [n.network for n in pop.networks])

    best_fit_log.append(max(fits)/TRAINING_SET_SIZE)
    pop_fit_log.append((sum(fits)/len(fits))/TRAINING_SET_SIZE)

    print(f'Best training accuracy: {best_fit_log[-1]*100:.2f}')

    best_net = pop.networks[fits.index(max(fits))].network
    network_details(best_net)

    # try it on validation set samples
    validation_fit = compute_fitness(best_net, validation_spikes, validation_labels, display_per_class=True, reconstruct_spikes=True)
    best_fit_validation_log.append(validation_fit/(VALIDATION_SET_SIZE))
    print(f'Validation set accuracy for best network: {validation_fit/(VALIDATION_SET_SIZE):.2f}')

    # write best network on validation set to file
    if len(best_fit_validation_log) == 1 or best_fit_validation_log[-1] > max(best_fit_validation_log[:-1]):
        print(f'Writing best network to {args.best_net_path}')
        with open(args.best_net_path, 'w') as f:
            json.dump(best_net.as_json(), f)

    # let's also save our fitness logs
    with open(args.log_path, 'a') as f:
        f.write(f'{best_fit_log[-1]},{best_fit_validation_log[-1]},{pop_fit_log[-1]}\n')

    print(f'This epoch took {time.time()-t0:.2f} to run')
    print('----------')
    pop = eons_inst.do_epoch(pop, fits, eons_param)