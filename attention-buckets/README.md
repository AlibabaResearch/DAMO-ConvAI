# Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use

The codes is implemented based on pytorch and codes for tool-use is base on [ToolBench](https://github.com/OpenBMB/ToolBench). We appreciate these open-source codes.

## Install
```bash
https://github.com/AlibabaResearch/DAMO-ConvAI.git
cd attention-buckets
conda create -n your_envs python=3.9
conda activate your_envs
pip install -r requirment.txt
# repalce "your_env_path/lib/site-packages/transformers/models/llama/modeling-llama.py" with our 'modeling-llama.py'
```
## Code for tool-use 
```bash
git clone git@github.com:OpenBMB/ToolBench.git
cd ToolBench
```
We only present information about our work, for more information about toolbench please refer to [ToolBench](https://github.com/OpenBMB/ToolBench)

### Data 
Put all dataset in ToolBench/data
- Original data of ToolBench: Download the dataset using the following link: [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/). 
- results of our method:  Download the dataset using the following link: [Google Drive](https://alibaba-research.oss-cn-beijing.aliyuncs.com/attention-buckets/all_data.zip)


### core codes and how to run and how to eval
1.replace "ToolBench/toolbench/inference/utils.py" with our "inference/utils.py"
2.move our "inference/config.py" to "ToolBench/toolbench/inference"
3.replace "ToolBench/toolbench/utils.py" with our "utils.py"
```bash
# run
bash scripts/inference_toolllama_pipeline.sh
# eval, the same to ToolBench
cd tooleval
bash run_convert_answer.sh
bash run_pass_rate.sh
# get pass rate of chatgpt_cot and your method and then run to get preference.
bash run_preference.sh 
```

## Code for rag
```bash
cd base_rag
```
### Data
Put the data in ../qa_dataset
Download the dataset using the following link: [Google Drive](https://alibaba-research.oss-cn-beijing.aliyuncs.com/attention-buckets/all_data.zip)

### how to run and eval
```bash
# run
# bsz >= total bases num
CUDA_VISIBLE_DEVICES=i python test_nq_kl.py --flag i  --bsz  8 --num_doc $num_doc --ngpu $n_gpu --data_name $data_name

# eval
python merge_result.py --ngpu $n_gpu --data_name $data_name --num_doc $num_doc
```


## Citation
Feel free to cite us if you like our work.
```bibtex
@article{Chen2023FortifyTS,
  title={Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use},
  author={Yuhan Chen and Ang Lv and Ting-En Lin and Chang Heng Chen and Yuchuan Wu and Fei Huang and Yongbin Li and Rui Yan},
  journal={ArXiv},
  year={2023},
  volume={abs/2312.04455},
  url={https://api.semanticscholar.org/CorpusID:266053571}
}
```

