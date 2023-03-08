# SPACE2.0
This repository contains code and data for the **COLING'2022** paper "**SPACE-2: Tree-Structured Semi-Supervised Contrastive Pre-training for Task-Oriented Dialog Understanding**".

Full version with Appendix is here: [[PDF]](https://arxiv.org/abs/2209.06638)

## Abstract
Pre-training methods with contrastive learning objectives have shown remarkable success in dialog understanding tasks. However, current contrastive learning solely considers the self-augmented dialog samples as positive samples and treats all other dialog samples as negative ones, which enforces dissimilar representations even for dialogs that are semantically related.  In this paper, we propose SPACE2.0, a tree-structured pre-trained conversation model, which learns dialog representations from limited labeled dialogs and large-scale unlabeled dialog corpora via semi-supervised contrastive pre-training. Concretely, we first define a general semantic tree structure (STS) to unify the inconsistent annotation schema across different dialog datasets, so that the rich structural information stored in all labeled data can be exploited.  Then we propose a novel multi-view score function to increase the relevance of all possible dialogs that share similar STSs and only push away other completely different dialogs during supervised contrastive pre-training. To fully exploit unlabeled dialogs, a basic self-supervised contrastive loss is also added to refine the learned representations. Experiments show that our method can achieve new state-of-the-art results on the DialoGLUE benchmark consisting of seven datasets and four popular dialog understanding tasks.

## Main Results
SPACE2.0 performs intent prediction, slot filling, semantic parsing and dialog state tracking, which achieves new state-of-the-art results on all seven benchmark datasets including: BANKING77, CLINC150, HWU64, REST8K, DSTC8, TOP and MultiWOZ2.1.

| Intent Prediction | BANKING77 | CLINC150 | HWU64 |
|:-------------------:|:-----:|:-----:|:-----:|
|       Accuracy      | 94.77 | 97.80 | 94.33 |

| Slot Filling | REST8K | DSTC8 |
|:-------------------:|:-----:|:-----:|
|       Macro-averaged F1 score      | 96.20 | 91.38 |

| Semantic Parsing | Exact-match |
|:-------------------:|:-----:|
|       TOP       | 82.74 |

| Dialog State Tracking | Joint Goal Accuracy |
|:-------------------:|:-----:|
|       MultiWOZ2.1       | 59.51 |

## Requirements
```
- torch == 1.8.0+cu111
- scikit-learn == 0.23.1
- numpy == 1.18.5
- nltk == 3.5
- spacy == 2.3.5
- scipy == 1.5.0
- regex == 2020.6.8
- tqdm == 4.60.0
- transformers == 2.9.1
```
We use the tokenization tool in SpaCy and you can directly install python packages by commands: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.

## Preparation
### Path Definition
Define your own paths `<YOUR_PROJECT_PATH>` and `<YOUR_SAVE_PATH>` in scripts as follows: 
```sh
PROJECT_NAME="SPACE2.0"  # project name (fixed)
PROJECT_ROOT=<YOUR_PROJECT_PATH>/${PROJECT_NAME}  # root path of this project
SAVE_ROOT=<YOUR_SAVE_PATH>/${PROJECT_NAME}  # root path of model's output
```

### Data Preparation
Download data-split1 from this [link](https://drive.google.com/file/d/1ocwnuOLxB3VzngeWZsm59IRrhEv22Scx/view?usp=share_link). 

The downloaded zip file `data.zip` contains pre-training corpora (including BANKING77, CLINC150 and HWU64) and three extra task-oriented (TOD) benchmark datasets: REST8K, DSTC8 and TOP, which have already been processed. You need to put the unzipped directory `data` into the project directory `SPACE2.0` for the subsequent training.

Download data-split2 from this [link](https://drive.google.com/file/d/1BZvlARzxXobjpQQRWvkF3jwnLN9-9c-n/view?usp=share_link).

The downloaded zip file `data.zip` contains one TOD benchmark dataset: MultiWOZ2.1, which have already been processed. You need to put the unzipped directory `data` into the directory `SPACE2.0/trippy` for the subsequent training.

### Directory Structure
Upon completion, your SPACE2.0 folder should contain the following:
```
SPACE2.0/
├── data  # pre-training corpora and downstream TOD datasets
├── model  # pre-trained checkpoints and vocabulary
├── outputs  # fine-tuned checkpoints
├── scripts  # inference bashes
├── space  # model and modules
├── tools  # misc tools
└── trippy  # separated code for dialog state tracking (dst)
    ├── data  # multiwoz2.1 datasets
    ├── dataset_config  # data configuration
    ├── model  # pre-trained checkpoint
    ├── outputs  # fine-tuned checkpoints for multiwoz2.1
    └── scripts  # inference bashes for multiwoz2.1
```

## Pre-training
### Pre-training Corpora
- [AnPreDial](https://drive.google.com/file/d/1ocwnuOLxB3VzngeWZsm59IRrhEv22Scx/view?usp=share_link): a new labeled dialog dataset annotated with semantic trees, which contains 32 existing labeled TOD datasets with 3
million turns, ranging from single-turn QA to multi-turn dialogs.
- [UnPreDial](https://drive.google.com/file/d/1ocwnuOLxB3VzngeWZsm59IRrhEv22Scx/view?usp=share_link): a large-scale unlabeled dialog dataset consisting of 19M utterances with careful processing from 21 online dialog corpora, ranging from online forums to conversational machine reading comprehension.

### Pre-trained Checkpoint
- [SPACE2.0](https://drive.google.com/file/d/1QOhrd_kB8VXevEAo1Gohr58LxMI4OjYo/view?usp=share_link): an uncased model (12-layers, 768-hidden, 12-heads, 110M parameters)

You need to unzip the downloaded model file `model.zip`, then put the unzipped directory `model` into the project directory `SPACE2.0` for the further fine-tuning.

### Training
We pre-train the SPACE2.0 on limited labeled dialogs (**AnPreDial**) and large-scale unlabeled dialog corpora (**UnPreDial**) via semi-supervised learning.
You can pre-train SPACE2.0 from scratch by running the following scripts:

```sh
# Step 1: Preprocess pre-training corpora
sh scripts/pre_train/preprocess.sh

# Step 2: Begin pre-training
sh scripts/pre_train/train.sh
```

## Fine-tuning
### Fine-tuned Checkpoints
Download checkpoints-split1 from this [link](https://drive.google.com/file/d/10QEEMNsjO5rH0ZRsJBj9zkDc5ozxc3Ch/view?usp=share_link). 

The downloaded zip file `outputs.zip` contains our best fine-tuned checkpoints on the following six datasets: 
- BANKING77, CLINC150, HWU64 (**Intent Prediction**)
- REST8K, DSTC8 (**Slot Filling**)
- TOP (**Semantic Parsing**)

If you want to reproduce our reported results, you should put the unzipped directory `outputs` into the directory `${SAVE_ROOT}` (set in scripts). 

Download checkpoints-split2 from this [link](https://drive.google.com/file/d/1G7K6AIBcRTC3CgMtSZdJ_TM6rFeXGe96/view?usp=share_link). 

The downloaded zip file `outputs.zip` contains our best fine-tuned checkpoints on one dataset: 
- MultiWOZ2.1 (**Dialog State Tracking**)

If you want to reproduce our reported results, you should put the unzipped directory `outputs` into the directory `SPACE2.0/trippy`.
Then you can directly run the inference scripts of different datasets for the reproduction, which will be introduced later.

### Training
We fine-tune the SPACE2.0 on the seven TOD datasets. You can fine-tune SPACE2.0 from scratch by running the following training scripts.

For intent prediction (**IP**) task:
```sh
# Step 1: Unsupervised Training on BANKING77, CLINC150 and HWU64
sh scripts/pre_train/train_intent.sh

# Step 2.1: Supervised Training on BANKING77
sh scripts/banking/train.sh

# Step 2.2: Supervised Training on CLINC150
sh scripts/clinc/train.sh

# Step 2.3: Supervised Training on HWU64
sh scripts/hwu/train.sh
```

For slot filling (**SF**) or semantic parsing (**SP**) task:
```sh
# Step 1: Unsupervised Training on REST8K, DSTC8, TOP and COCO
sh scripts/pre_train/train_slot.sh

# Step 2.1: Supervised Training on REST8K
sh scripts/rest8k/train.sh

# Step 2.2: Supervised Training on DSTC8
sh scripts/dstc8/train.sh

# Step 2.3: Supervised Training on TOP
sh scripts/top/train.sh
```

For dialog state tracking (**DST**) task:
```sh
# Training on MultiWOZ2.1
cd SPACE2.0/trippy/
sh scripts/multiwoz21/train.sh
```

> **NOTE**: You can skip Step 1 if you directly download the output model of Step 1. 
> For DST task, you should convert model parameters into Hugging Face format.
> So you can download the model file from this [link](https://drive.google.com/file/d/1xzKhKBg0hJPAq1NebluLIwfVxnfN1-1R/view?usp=share_link) directly.
> Then you need to unzip the downloaded model file `model.zip`, and put the unzipped directory `model` into the directory `SPACE2.0/trippy` for the further fine-tuning.

### Inference
After collecting some fine-tuned checkpoints (by directly using ours or fine-tuning SPACE2.0 from scratch by yourself), you can do the inference on the test sets of these datasets by running the following inference scripts.

For intent prediction (**IP**) task:
```sh
# Inference on BANKING77
sh scripts/banking/infer.sh

# Inference on CLINC150
sh scripts/clinc/infer.sh

# Inference on HWU64
sh scripts/hwu/infer.sh
```

For slot filling (**SF**) or semantic parsing (**SP**) task:
```sh
# Inference on REST8K
sh scripts/rest8k/infer.sh

# Inference on DSTC8
sh scripts/dstc8/infer.sh

# Inference on TOP
sh scripts/top/infer.sh
```

For dialog state tracking (**DST**) task:
```sh
# Inference on MultiWOZ2.1
cd SPACE2.0/trippy/
sh scripts/multiwoz21/infer.sh
```

> **NOTE**: For reproduction, all the best hyper-parameters have already been set in corresponding scripts and you can follow them to run.

## References
- For the data preparation and evaluation on BANKING77/CLINC150/HWU64/REST8K/DSTC8/TOP, we refer to the code of [DialoGLUE](https://github.com/alexa/dialoglue).
- For the data preparation and evaluation on MultiWOZ2.1, we refer to the code of [TripPy](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public).

## Citation
If you use our code or find SPACE-2 useful in your work, please cite our paper as:

```bib
@inproceedings{he2022tree,
  title={SPACE-2: Tree-Structured Semi-Supervised Contrastive Pre-training for Task-Oriented Dialog Understanding},
  author={He, Wanwei and Dai, Yinpei and Hui, Binyuan and Yang, Min and Cao, Zheng and Dong, Jianbo and Huang, Fei and Si, Luo and Li, Yongbin},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  year={2022}
}
```

## Contact
For personal communication related to SPACE-2, please contact Wanwei He (`ww.he@siat.ac.cn`).
