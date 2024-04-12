eval "$(conda shell.bash hook)"
conda activate mlaudio
python snn.py\
	-a train \
	--proc_params ./config/risp.json\
	--eons_params ./config/eons.json\
	--timeseries true \
	--app_type load \
	--data_np /data2/khood/GitHub/MLAudio/numpyData/valid.npy \
	--labels_np /data2/khood/GitHub/MLAudio/numpyData/valid_labels.npy \
	--encoder config/encoder.json \
	--processes 1 \
	--sim_time 7201 \
	--epochs 3 \
	--network_filename test_eons.json\
    --split 0
