# DialSTART
This repository contains the code and data for our paper Unsupervised Dialogue Topic Segmentation with
Topic-aware Utterance Representation ([DialSTART](http://arxiv.org/abs/2305.02747))

## Installation
```commandline
pip3 -r install requirements.txt
```
## Data
Please first download and unzip the Dialseg711 and Doc2dial data from [here](https://drive.google.com/drive/folders/1Ttd83n2KMGErrioDDruuQonaTuHMFGMR?usp=share_link).
And put the unziped data into `data/`
To get the preprocessed data, please run following code:
```commandline
python data_preprocess.py
```

## Model
Our model are availble [here](https://drive.google.com/drive/folders/1HhN_Gr3vgX6haFLCjIzainvBPVl40MNK?usp=sharing).
Please download it and move into `model/`

## Train
To train our model on the preprocessed data, please run following code:
```commandline
sh scripts/train.sh
```

## Inference
To reproduce our result in the paper, please run following code:
```
scripts/test.sh
```

