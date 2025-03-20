[//]: # (#Unified Data-to-Text Pretraining)


## Unified Structured Data as Graph for Data-to-Text Pretraing

## Prepare Environment
You can create an environment for UniD2T and directly install python packages by commands: 
```
pip install -r requirements.txt 
```


## Data_preprocess
You can download the original data from the original website:
[ToTTo](https://github.com/google-research-datasets/ToTTo),
[CoSQL](https://yale-lily.github.io/cosql),
[WebNLG](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0),
[DART](https://github.com/Yale-LILY/DART),
[WikiBio](https://rlebret.github.io/wikipedia-biography-dataset/),
[WikiTableT](https://github.com/mingdachen/WikiTableT).

Then put it in the ```/orig_datasets/``` directory and use the code in ```/data_preprocess/``` to process each data. The processed data will be saved in cleanout_datasets, such as in totto dataset:
```
python /data_preprocess/totto/convert_totto_to_unified_graph.py
```

## Pretrain
Merge the data processed in the previous step:
```
python /data_preprocess/convert_totto_to_unified_graph.py
```
Pre-training on multiple GPUs:
```
torchrun \
--nproc_per_node=4 \
./pretrain.py \
--config /pretrain_config/**.yml
```


## Fintune
Fintune on single GPUs:
```
python finetune.py --config config/**.yml
```




