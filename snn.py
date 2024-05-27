from py_apps.utils.common_utils import read_network
from py_apps.utils.common_utils import load_json_arg
from py_apps.utils.neuro_help import *
from models import EONSClassifyAudioApp


def main():
    parser = argparse.ArgumentParser(description="Classification Application Driver")
    parser.add_argument("--activity", "-a", required=True, type=str, choices=["train", "test"], help="activity to perform")
    parser.add_argument("--network_filename", default=None, type=str, help="location to store the best network file produced if training or network to load if testing")
    parser.add_argument("--sim_time", default=50, type=float, help="the simulation timefor each data instance")
    parser.add_argument("--eons_params", default="config/eons.json", type=str, help="JSON file with EONS parameters")
    parser.add_argument("--extra_eons_params", default="{}", type=str ,help="JSON file or JSON string updating EONS parameters from configuration file")
    parser.add_argument("--epochs", default=50, type=int, help="epochs for eons")
    parser.add_argument("--max_fitness", default=1e+06, type=float, help="max fitness for eons")
    parser.add_argument("--processes", default=1, type=int, help="processes for EONS")
    parser.add_argument("--test_seed", default=1234, type=int, help="testing seed")

    print("Reading classification params")
    add_class_arguments(parser)

    print("Reading proc params")
    add_proc_arguments(parser)

    print("Reading encoder information")
    add_coder_arguments(parser)

    print("Reading printing params")
    add_printing_arguments(parser)

    args = parser.parse_args()

    print("Reading in proc params as necessary")
    proc_params = load_json_arg(args.proc_params)
    extra_proc_params = load_json_arg(args.extra_proc_params)
    for k in extra_proc_params.keys():
        proc_params[k] = extra_proc_params[k]

    print("Reading in EONS params")
    eons_params = load_json_arg(args.eons_params)

    print("Reading in extra EONS params")
    extra_eons_params = load_json_arg(args.extra_eons_params)

    for k in extra_eons_params.keys():
        eons_params[k] = extra_eons_params[k]

    proc_instantiation = get_proc_instantiation(args.proc_name)

    print("Instantiate data")
    X, y = setup_data(args)

    config = setup_class_config(args)
    print("Starting training")
    if (args.activity == "train"):
        app = EONSClassifyAudioApp(config, X, y)

        train_params = {}
        train_params["eons_params"] = eons_params
        train_params["num_epochs"] = args.epochs
        train_params["num_processes"] = args.processes
        train_params["fitness"] = args.fitness
        train_start = time.time()
        app.train(train_params, proc_instantiation, proc_params)
        train_end = time.time()
        elapsed = train_end-train_start
        net = app.overall_best_net
        net.prune()
        print("Network size:", net.num_nodes(), net.num_edges())
        print("Training Time:", elapsed)

if __name__ == "__main__":
    main()