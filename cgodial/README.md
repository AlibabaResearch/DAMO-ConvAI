# Chinese Goal-oriented Dialog (CGoDial)  

This is a new challenging and comprehensive Chinese benchmark for multi-domain Goal-oriented Dialog evaluation, which covers three datasets with different knowlwdge soueces: slot-based dialog, Flow-based Dialog and Retrieval-based Dialog.  

The datases is in the [google drive](https://drive.google.com/file/d/1_CDFgcpFVo4KJJIFv4P1xGfpg0RjFQLd/view?usp=sharing). Please download the datasets and merge the datasets with the codes in the git by name of the path.  

## Slot-based Dialog  
`cd slot_based_dialog`  
The datasets is in `./data`, there are two baselines:  
1. Chinese gpt, download the [model](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) and put it in the dir `cdial_gpt` and go to the path, run the `run.sh` to train and test, and use `eval.py` to get the evaluation results  
2. Chinese T5, download the [model](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) and put it in the dir `chinese_t5` and go to the path, run `run.sh` for train and test, and use `eval.py` to get the evaluation results

## Flow-based Dialog
`cd flow_based_dialog`  
The datasets is in `./data`, there are two baselines:  
1. Roberta-wwm, download the [model](https://huggingface.co/uer/roberta-base-wwm-chinese-cluecorpussmall)  
2. StructBERT, download the [model](https://github.com/alibaba/AliceMind/tree/main/StructBERT)  
use the `run.sh` for training (set is_train) or test (set is_eval) and get the json output file, and run the `eval.py` for the result  

## Retrieval_based Dialog 
`cd retrieval_based_dialog`  
The datasets is `train.json, dev.json, test.json`  
ues the same two baseline models and codes with Flow-based Dialog  
use the `run.sh` for training (set is_train) or test (set is_eval) and get the json output file, and run the `ECDMetric.py` for the result.

## Citation  
You can cite our paper with the information:  
```
@article{dai2022cgodial,
  title={CGoDial: A Large-Scale Benchmark for Chinese Goal-oriented Dialog Evaluation},
  author={Dai, Yinpei and He, Wanwei and Li, Bowen and Wu, Yuchuan and Cao, Zheng and An, Zhongqi and Sun, Jian and Li, Yongbin},
  journal={arXiv preprint arXiv:2211.11617},
  year={2022}
}
```
