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
from multiprocessing import shared_memory
import signal
import time
import functools
import json
from common import risp_config, read_spikes_from_disk, network_details

# TODO: also, look into how we are reading filenames to ensure randomness

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--num_processes', required=True)
parser.add_argument('-d', '--dataset_path', required=True)
parser.add_argument('-b', '--best_net_path', required=True) # json for best network path
parser.add_argument('-l', '--log_path', required=True)
parser.add_argument('-e', '--epoch_count', required=True)
parser.add_argument('--synapse_count', required=True)
parser.add_argument('--hidden_count', required=True)
parser.add_argument('--num_mutations', required=True)
parser.add_argument('--random_factor', default='0.10', required=False)
parser.add_argument('--mutation_rate', default='0.9', required=False)
parser.add_argument('--mutations_weights', default='0.75/0.25/0.75/0.25/2.5/2.5/0', required=False, 
    help='7 \'/\' separated values for mutation weights, order is: add_node, delete_node, add_edge, delete_edge, node_params, edge_params, net_params')
parser.add_argument('--edge_mutations', default='0.65/0.35', required=False,
    help='2 \'/\' separated values for edge mutations eons param, order is: weight, delay')
parser.add_argument('--inject', required=False,
    help='Path to JSON file of network to inject into population')
parser.add_argument('--mode', default='s2s', choices=['s2s', 'samples'], required=False)
parser.add_argument('--proc_timesteps', default='1000', required=False)

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

MODE = args.mode

if MODE == 's2s':
    NUM_INPUT_NEURONS = 80
elif MODE == 'samples':
    NUM_INPUT_NEURONS = 1

NUM_OUTPUT_NEURONS = 2
NUM_SYNAPSES = int(args.synapse_count)
NUM_HIDDEN_NEURONS = int(args.hidden_count)
POP_SIZE = 70

MOA = neuro.MOA()
MOA.seed(23456789, '')

NUM_PROCESSES = int(args.num_processes)
PROC_RUN_TIMESTEPS = int(args.proc_timesteps)

# Configure EONS
mut_weight_values = args.mutations_weights.split('/')
if len(mut_weight_values) != 7:
    print('Invalid input for mutations_weights (see default value in script, and check -h)')
    exit()

mut_weight_values = [float(w) for w in mut_weight_values]

edge_mutations_values = args.edge_mutations.split('/')
if len(edge_mutations_values) != 2:
    print('Invalid input for edge_mutations (see default value in script, and check -h)')
    exit()

edge_mutations_values = [float(v) for v in edge_mutations_values]

eons_param = {
    "starting_nodes": NUM_HIDDEN_NEURONS,
    "starting_edges": NUM_SYNAPSES,
    "merge_rate": 0,
    "population_size": POP_SIZE,
    "multi_edges": 0,
    "crossover_rate": 0.9,
    "mutation_rate": float(args.mutation_rate),
    "selection_type": "tournament",
    "tournament_size_factor": 0.1,
    "tournament_best_net_factor": 0.9,
    "random_factor": float(args.random_factor),
    "num_mutations": float(args.num_mutations),
    "node_mutations": { "Threshold": 1.0 },
    "net_mutations": { },
    "edge_mutations": { "Weight": edge_mutations_values[0], "Delay": edge_mutations_values[1],  },
    "num_best" : 2,
    "add_node_rate": mut_weight_values[0],
    "delete_node_rate": mut_weight_values[1],
    "add_edge_rate": mut_weight_values[2],
    "delete_edge_rate": mut_weight_values[3],
    "node_params_rate": mut_weight_values[4],
    "edge_params_rate": mut_weight_values[5],
    "net_params_rate" : mut_weight_values[6]
}
print(eons_param)

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

def compute_fitness(net, spikes_shm_name, labels, spikes_shm_dtype, spikes_shm_shape, display_per_class=False, reconstruct_spikes=False):
    # get shared memory values
    shm_spikes = shared_memory.SharedMemory(name=spikes_shm_name)
    shared_spikes_arr = np.ndarray(shape=spikes_shm_shape, dtype=spikes_shm_dtype, buffer=shm_spikes.buf)

    if reconstruct_spikes:
        rec_spikes = []

        if MODE == 's2s':
            for i in range(shared_spikes_arr.shape[0]): # sample
                rec_spikes.append([])
                
                for j in range(shared_spikes_arr.shape[1]): # channel
                    rec_spikes[i].append([])
                    
                    for k in range(shared_spikes_arr.shape[2]): # timestep
                        if shared_spikes_arr[i][j][k][2] == 1: # value component
                            rec_spikes[i][j].append(neuro.Spike(
                                shared_spikes_arr[i][j][k][0], shared_spikes_arr[i][j][k][1], shared_spikes_arr[i][j][k][2]))

        elif MODE == 'samples':
            # for this one our dimensions are: (data sample #, 24k audio samples)
            for i in range(shared_spikes_arr.shape[0]): 
                rec_spikes.append([])
                
                for j in range(shared_spikes_arr.shape[1]):
                    # id, time, value
                    rec_spikes[i].append(neuro.Spike(0, j, shared_spikes_arr[i][j]))



        spikes = rec_spikes

    shm_spikes.close()
    
    proc.load_network(net)

    correct = 0
    per_class = {0: 0, 1: 0}
    for i in range(len(spikes)):
        proc.clear_activity()

        if MODE == 's2s':
            for c in spikes[i]: # spikes[i] is a single training sample
                proc.apply_spikes(c)
        elif MODE == 'samples':
            proc.apply_spikes(spikes[i])

        proc.run(PROC_RUN_TIMESTEPS)

        out_counts = proc.output_counts()

        prediction = 0 if out_counts[0] > out_counts[1] else 1

        if prediction == labels[i]:
            correct += 1
            per_class[prediction] += 1

    if display_per_class:
        print(f'For class 0: {per_class[0]}/{(labels == 0).sum()}')
        print(f'For class 1: {per_class[1]}/{(labels == 1).sum()}')

    return correct

def load_network(path):
    loaded_net = neuro.Network()
    with open(path, 'r') as f:
        network_json = json.load(f)
        loaded_net.from_json(network_json)

    return loaded_net


EPOCH_COUNT = int(args.epoch_count)
eons_inst.set_template_network(template_net)
pop = eons_inst.generate_population(eons_param, MOA.random_integer())

best_fit_log = []
best_fit_validation_log = []
pop_fit_log = [] # mean
TRAINING_SET_SIZE = training_labels.shape[0]
VALIDATION_SET_SIZE = validation_labels.shape[0]
t0, t1 = 0, 0

# inject into pop
if args.inject is not None:
    pop.replace_network(1, load_network(args.inject))

# create shared memory - TODO: make this good
# train set shared memory
training_spikes_arr = np.array(training_spikes)
train_spikes_shape = training_spikes_arr.shape
train_spikes_dtype = training_spikes_arr.dtype
train_shm_size = train_spikes_dtype.itemsize * np.prod(train_spikes_shape)

shm = shared_memory.SharedMemory(create=True, size=train_shm_size)
shm_name = shm.name

shared_arr = np.ndarray(dtype=train_spikes_dtype, shape=train_spikes_shape, buffer=shm.buf)
shared_arr[:] = training_spikes_arr[:]

# validation set shared memory
validation_spikes_arr = np.array(validation_spikes)
validation_spikes_shape = validation_spikes_arr.shape
validation_spikes_dtype = validation_spikes_arr.dtype
validation_shm_size = validation_spikes_dtype.itemsize * np.prod(validation_spikes_shape)

validation_shm = shared_memory.SharedMemory(create=True, size=validation_shm_size)
validation_shm_name = validation_shm.name

shared_val_arr = np.ndarray(dtype=validation_spikes_dtype, shape=validation_spikes_shape, buffer=validation_shm.buf)
shared_val_arr[:] = validation_spikes_arr[:]

# training loop
for i in range(EPOCH_COUNT):
    print(f'Starting epoch {i}...')
    t0 = time.time()

    compute_fitness_partial = functools.partial(compute_fitness, spikes_shm_name=shm_name, labels=training_labels,
                                                spikes_shm_shape=train_spikes_shape,
                                                spikes_shm_dtype=train_spikes_dtype,
                                                reconstruct_spikes=True)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        fits = p.map(compute_fitness_partial, 
                    [n.network for n in pop.networks])

    best_fit_log.append(max(fits)/TRAINING_SET_SIZE)
    pop_fit_log.append((sum(fits)/len(fits))/TRAINING_SET_SIZE)

    print(f'Best training accuracy: {best_fit_log[-1]*100:.2f}')

    best_net = pop.networks[fits.index(max(fits))].network
    network_details(best_net)

    # try it on validation set samples
    validation_fit = compute_fitness(best_net, validation_shm_name, validation_labels,
                                     validation_spikes_dtype,
                                     validation_spikes_shape,
                                     display_per_class=True, reconstruct_spikes=True)    
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

shm.close()
shm.unlink()
validation_shm.close()
validation_shm.unlink()