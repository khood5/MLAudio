eval "$(conda shell.bash hook)"
conda activate mlaudio
python eons.py\
	-a train \
	--proc_params ./config/risp.json\
	--eons_params ./config/eons.json\
	--timeseries true \
	--app_type load \
	--data_np debug.npy \
	--labels_np debug_labels.npy \
	--encoder config/encoder.json \
	--processes 16 \
	--sim_time 2401 \
	--epochs 1000 \
	--network_filename firstRun.json\
    --split 0