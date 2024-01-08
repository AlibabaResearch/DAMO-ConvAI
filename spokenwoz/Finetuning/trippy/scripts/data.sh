tar -zxvf ../audio/audio_5700_train_dev.tar.gz

python ./dataset.py --data_name data.json \
--root ./data \
--audio_path ../audio/audio_5700_train_dev \
--output_path ./data

python ./utils_dst.py --data_root ./data


