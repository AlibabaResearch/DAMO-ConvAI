## Installation
```commandline
pip install -r requirements.txt
```
If you want to use apex for AMP training, please clone the apex source code from the repository at github.com to install.

## Fine-tuning
We provide the pre-trained checkpoint of our model at [huggingface.co](https://huggingface.co/publicstaticvo/SPECTRA-base). To reproduce our result in the paper, please first download the pre-processed fine-tuning data (be available soon), then run `scripts/finetune.sh`
### Datasets
Here are the processed fine-tuning data datasets: 
[**MOSI**](https://space-mm-data.oss-cn-wulanchabu.aliyuncs.com/downstreamv2/mosi.tgz), 
[**MOSEI**](https://space-mm-data.oss-cn-wulanchabu.aliyuncs.com/downstreamv2/mosei.tgz), 
[**IEMOCAP**](https://space-mm-data.oss-cn-wulanchabu.aliyuncs.com/downstreamv2/iemocap.tgz), and 
[**MINTREC**](https://space-mm-data.oss-cn-wulanchabu.aliyuncs.com/downstreamv2/mintrec.tgz). 
These are all composed by pickles and can be used directly. 

> Due to the large data size of SpokenWOZ and Spotify-100k (tens of GBs), please obtain from the original repo."

### Usage
To access the training, validation, and test files in the datasets, you can use the following command to extract the mosi.tgz file:

```
tar -xzvf mosi.tgz
```

Once extracted, you'll find .pkl files for training, validation, and testing. Each pickle file contains a list of samples, and each sample includes the following components:
1. Audio Features: This field contains the audio feature data.
2. Text Token IDs: Here, you'll find the IDs corresponding to text tokens.
3. Label: This is the label assigned to the sample.
4. History Audio Features (if applicable): If present, this field contains historical audio feature data.
5. History Text Token IDs (if applicable): Similar to the above, this includes historical text token IDs, if available.

We hope this information helps you in utilizing the dataset effectively. Should you have any questions or need further assistance, please feel free to reach out.

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
