## Installation
```commandline
pip install -r requirements.txt
```
If you want to use apex for AMP training, please clone the apex source code from the repository at github.com to install.

## Fine-tune
We provide the pre-trained checkpoint of our model at [huggingface.co](https://huggingface.co/publicstaticvo/SPECTRA-base). To reproduce our result in the paper, please first download the pre-processed fine-tuning data (be available soon), then run `scripts/finetune.sh`

## Pre-Train
To pretrain our model from scratch, please first download our processed pretraining dataset (be available soon), then download pre-trained WavLM and RoBERTa models from huggingface.co (optional), and run `scripts/train-960.sh`

<!--
```commandline
python run_dst.py --model spectra --model_type roberta \
    --data_dir ./data \
    --model_dir /PATH/OF/YOUR/PRETRAINED/SPECTRA/MODEL \
    --output_dir ./result \
    --dataset_config ./data/spokenwoz_config.json \
    --per_gpu_train_batch_size 2 \
    --accum 4
```
-->
