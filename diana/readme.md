# Diana
The PyTorch implementation of paper [Domain Incremental Lifelong Learning in an Open World](https://arxiv.org/abs/2305.06555) (ACL 2023)

## Requirements
```bash
cd DomainIncrementalLL
pip install -r r.txt
```
## QA tasks
### Data preparation
Download datasets, unzip and place in /data_process

[QA Dataset](https://drive.google.com/file/d/1x8vvrdfwKCXpT_M4NA_eso8cX-LlMWsQ/view?usp=share_link)

Tokenize plain text datasets:
```python
cd downstream
python create_l2p.py --model_name_or_path t5-base --dataset_name null --output_dir null
```

### Train and evaluate on 4 GPUs
```python
bash ./diana.sh ./ll 8 48 
```

### ablations: No task prompt & No meta prompt

To evaluate ablation without task prompts:
```bash
bash ./dianawotask.sh ./ll 8 48
```
To evaluate ablation without meta prompts:
```bash
bash ./dianawometa.sh ./ll 8 48
```

## DecaNLP tasks

[Dataset](https://drive.google.com/file/d/1pzIoTzooaecsU4HXP_n4-_7Uuxv9i4A-/view?usp=share_link)

extract plain texts from raw files:
```bash
cd downstreamdeca
python extract.py
```
tokenize plain texts:
```bash
python toknize.py
```


Run the training and evaluation script:
```bash
bash ./diana.sh ./ll 8 48
```
