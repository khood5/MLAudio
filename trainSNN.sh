eval "$(conda shell.bash hook)"
conda activate mlaudio
python snn.py\
	-a train \
	--proc_params ./config/risp.json\
	--eons_params ./config/eons.json\
	--timeseries true \
	--app_type load \
	--data_np neuroTrain.small.npy \
	--labels_np neuroTrain.small_labels.npy \
	--encoder config/encoder.json \
	--processes 16 \
	--sim_time 2401 \
	--epochs 1000 \
	--network_filename firstRun.json\
    --split 0