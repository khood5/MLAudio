import subprocess
import os

run_params = [(2200+i*150, 25+i*25) for i in range(20)]
CORES_PER_PROCESS = 5
DATASET_PATH = "./data/dataset-200.npz" 
OUT_DIR = './runs3/?/' # replace ? with run number

for i, params in enumerate(run_params):
    out_path = OUT_DIR.replace('?', str(i))

    os.mkdir(out_path)
    print(f'Created dir at {out_path}')

    out_log = open(f'{out_path}out.txt', 'w')

    print(f'Running train_script.py with {params[0]} synapses and {params[1]} hidden neurons')
    process = subprocess.Popen([
            "python3", "train_script.py",
            "-p", f"{CORES_PER_PROCESS}",
            "-d", f"{DATASET_PATH}",
            "-b", f"{out_path}best_net.json",
            "-e", "3",
            "-l", f"{out_path}1200-test-log.csv",
            "--hidden_count", f"{params[1]}",
            "--synapse_count", f"{params[0]}",
            "--num_mutations", "20"
    ], stdout=out_log)

    out_log.close()

    process.wait()