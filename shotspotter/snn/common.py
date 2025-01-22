import numpy as np
import json

risp_config = {
  "min_weight": -1,
  "max_weight": 1,
  "min_threshold": -1,
  "max_threshold": 1,
  "min_potential": -1,
  "max_delay": 10,
  "discrete": False
}

def read_spikes_from_disk(path):
    data = np.load(path)
    return data['train_set'], data['train_labels'], data['validation_set'], data['validation_labels'], data['test_set'], data['test_labels']


def network_details(nw, log_json=False):
    net_json = nw.as_json()

    if log_json:
        print(net_json)
    
    print(f'Network has {len(net_json["Edges"])} synapses and {len(net_json["Nodes"])} neurons')

    # check if all out nodes have an incoming synapse
    out_ids = net_json['Outputs']
    for edge in net_json['Edges']:
        if edge['to'] in out_ids:
            out_ids.remove(edge['to'])

    if len(out_ids) == 0:
        print('All outputs have incoming connections')
    else:
        print(f'Outputs {out_ids} have no incoming connections')
