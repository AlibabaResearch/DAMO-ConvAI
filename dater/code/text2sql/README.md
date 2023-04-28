# Binder🔗: Binding Language Models in Symbolic Languages
<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/HKUNLP/Binder?color=green">
        <img src="https://img.shields.io/github/last-commit/HKUNLP/Binder?color=green">
    </a>
    <br/>
</p>

Code for paper [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875). 
Please refer to our [project page](https://lm-code-binder.github.io/) for more demonstrations and up-to-date related resources. 
Check out our [demo page](https://huggingface.co/spaces/hkunlp/Binder) to have an instant experience of Binder, which achieves **sota or comparable performance with only dozens of(~10) program annotations**.

<img src="pics/binder.png" align="middle" width="100%">

## Updates
- **2022-12-04**: Due to the fact OpenAI's new policy on request limitation, the n sampling couldn't be done as previously, we will add features to call multiple times to be the same usage soon!
- **2022-10-06**: We released our [code](https://github.com/HKUNLP/binder), [huggingface spaces demo](https://huggingface.co/spaces/hkunlp/Binder) and [project page](https://lm-code-binder.github.io/). Check it out!

## Dependencies
To establish the environment run this code in the shell:
```bash
conda env create -f py3.7binder.yaml
pip install records==0.5.3
```
That will create the environment `binder` we used.


## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate text2sql
``````

### Add key
Apply and get `API keys`(sk-xxxx like) from [OpenAI API](https://openai.com/api/), save the key in `key.txt` file, make sure you have the rights to access the model(in the implementation of this repo, `code-davinci-002`) you need.

### Run
Check out commands in `run.py`

## Citation
If you find our work helpful, please cite as
```
@article{Binder,
  title={Binding Language Models in Symbolic Languages},
  author={Zhoujun Cheng and Tianbao Xie and Peng Shi and Chengzu Li and Rahul Nadkarni and Yushi Hu and Caiming Xiong and Dragomir Radev and Mari Ostendorf and Luke Zettlemoyer and Noah A. Smith and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.02875}
}
```

## Contributors
<a href="https://github.com/BlankCheng">  <img src="https://avatars.githubusercontent.com/u/34505296?v=4"  width="50" /></a> 
<a href="https://github.com/Timothyxxx">  <img src="https://avatars.githubusercontent.com/u/47296835?v=4"  width="50" /></a>
<a href="https://github.com/chengzu-li"><img src="https://avatars.githubusercontent.com/u/69832207?v=4"  width="50" /></a>
<a href="https://github.com/Impavidity">  <img src="https://avatars.githubusercontent.com/u/9245607?v=4"  width="50" /></a> 
<a href="https://github.com/Yushi-Hu"><img src="https://avatars.githubusercontent.com/u/65428713?v=4"  width="50" /></a>
<a href="https://github.com/taoyds"><img src="https://avatars.githubusercontent.com/u/14208639?v=4"  width="50" /></a>


