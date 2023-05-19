# Large Language Models are Versatile Decomposers:Decomposing Evidence and Questions for Table-based Reasoning

The official repository which contains the prompt and the generation results of Codex at each stage for our paper [D<span style="font-size:0.8em;">a</span>t<span style="font-size:0.8em;">er</span>: Large Language Models are Versatile Decomposers: Decomposing Evidence and Questions for Table-based Reasoning](https://arxiv.org/pdf/2301.13808.pdf).
## Overview
In this study, we present a new method called **Dater** , which involves the decomposition of large tables of evidence into smaller sub-tables and the decomposition of complex questions into simpler sub-questions for text reasoning. Additionally, we introduce a novel "parsing-execution-filling" strategy to alleviate the issue of hallucination in Language Language Models (LLMs).
![Overview](./static/images/dater_animation.gif)




## Download
Download required prompts and saved files and moving files to target folder.

### Step 1
Download the [saved files and prompts](https://bird-bench.oss-cn-beijing.aliyuncs.com/dater_saved.tar.gz).

### Step 2
Move saved files to target folder.

```
tar -zxvf saved.tar.gz
sh mv_data2path.sh
```


## Evaluation
`sh run_{}.sh` 
can be easily used to evaluate our methods. If you want to obtain request results from the OpenAI API on your own, you will need to install a specific environment

## Environment
We suggest using Conda to set up the environment:
```
conda env create -f py3.7text2sql.yaml
pip install records==0.5.3
```
<!-- Please modify the control variables in the .sh files, such as decompose, parsing_execution_filling, reasoning, etc." -->

## Citation
If our work is useful for you, please consider citing our paper:



```bibtex
@inprocessing{ye2023large,
  author    = {Yunhu Ye and Binyuan Hui and Min Yang and Binhua Li and Fei Huang and Yongbin Li},
  title     = {Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning},
  booktitle = {SIGIR},
  year      = {2023},
}
```


## Acknowledgement

This implementation is based on [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875).
Our work is also thanks to [PASTA: Table-Operations Aware Fact Verification via Sentence-Table Cloze Pre-training](https://arxiv.org/abs/2211.02816).
Thanks to the author for releasing the code.