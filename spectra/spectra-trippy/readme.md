
## Data
Please first download and unzip the spokenwoz data.(Be available soon)
And put the unziped data into `data/`
To get the preprocessed data, please run following code:
```commandline
sh scripts/data.sh
```

## Train
To train our model on the preprocessed data, please run following code:
```commandline
python run_dst.py --model spectra --model_type roberta \
    --data_dir ./data \
    --model_dir /PATH/OF/YOUR/PRETRAINED/SPECTRA/MODEL \
    --output_dir ./result \
    --dataset_config ./data/spokenwoz_config.json \
    --per_gpu_train_batch_size 2 \
    --accum 4
```

## Inference
To reproduce our result in the paper, please run following code:
```
python run_dst.py --model spectra --model_type roberta \
    --data_dir ./data \
    --model_dir ./model \
    --output_dir ./result \
    --dataset_config ./data/spokenwoz_config.json \
    --evaluate
```

