

<div align="center">
<h1 align="center"> 🌊 FlowBench 🌊</h1>
<b>FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents</b>

<p align="center"><font size=6>📃</font> <a target="_self" href="https://arxiv.org/abs/2406.14884"> <img style="height:14pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a> <font size=6>•</font> <font size=6>🔔</font> <a target="_self" href="https://github.com/Justherozen/FlowBench"> <img style="height:14pt" src="https://img.shields.io/badge/-Code-pink?style=flat&logo=github"></a></p>

</div>


## Overview

This repository contains the source data and code for our EMNLP 2024 paper [FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents](https://arxiv.org/abs/2406.14884).  We propose a comprehensive benchmark, FlowBench, for workflow-guided agent planning. We first revisit and formalize different workflow knowledge formats for agent planning. FlowBench covers an extensive taxonomy (6 domains, 22 roles, 51 scenarios) and different knowledge formats (text, code, flowchart) to synchronize with real-world applications comprehensively. The benchmark data is constructed through a three-phase pipeline of task collection, workflow organization, and session generation. FlowBench features numerous distinct characteristics, such as coverage, difficulty, expertlevel annotation, and support for multi-round useragent interaction. Through extensive experiments on FlowBench, we find that even the best-performing model, GPT4o, fails to deliver satisfying results on challenging FlowBench. We hope that our work can provide meaningful insights to future research in the field of workflow-guided agent planning. An overview of our proposed FlowBench can be seen as follows:

![overview of flowbench](./resources/flowbench.png)

> *Please find more details of this work in our paper.*







### Dataset Introduction

Download `turn_data.zip` and `session_data.zip` from [Google Drive](https://drive.google.com/drive/folders/1PFzA5e-fuKpVZvAHP-otBhWPdU60O3d4?usp=sharing). After extracting, you will get two folders: `turn_data` and `session_data`. Move these two folders into the `data` directory. There two folders contains the benchmark data on the session-level and turn-level. All workflow knowledge with different formats has been organized into the `knowledge.json`.





### Evaluating workflow-guided agent planning

##### Dependencies

To install requirements:

	pip install requirements.txt

##### API preparation

Set up your OPENAI key in ./utils/keys.json

```
api_key: "Your OPENAI key"
```

After that, you can conduct the turn-level and session-level evaluations. 

##### Turn-level evaluation

- To generate the single-turn predictions for different test samples, please run

```
python ./turn_level/turn_inference.py --input_path INPUT_FOLDER --output_path OUTPUT_FOLDER
```

* Then you can calculate and display the evaluation metrics with the following commands, where `OUTPUT_FOLDER`  is the output  path of the last generation step.

```
 python ./turn_level/turn_metric_display.py --output_path OUTPUT_FOLDER
```



##### Session-level evaluation

- To simulate the predicted sessions, use the following commands with simulate mode, where `INPUT_PATH`, `OUTPUT_PATH`, and `EVAL_PATH` indicate the paths for test input, simulation generation, and simulation evaluation results, respectively.

```
python ./session_level/session_simulate.py --mode simulate --input_path INPUT_PATH --output_path OUTPUT_PATH --eval_path EVAL_PATH 
```

* After session simulation, you can calculate and save the evaluation metrics using the eval mode as follows.

```
python ./session_level/session_simulate.py --mode eval --input_path INPUT_PATH --output_path OUTPUT_PATH --eval_path EVAL_PATH 
```

* Finally, you can display the evaluation metrics for each scenario and optionally save them to excel file.
```
python ./session_level/session_metric_display.py --eval_path EVAL_PATH
```

You can specify the LLM used for generation, the LLM used as a judge, and the LLM used for environment simulation from the command line.




##### Future plans

We will keep refining our benchmark quality and our evaluation framework as part of our future initiatives!



### Citation

If you use or extend our work, please cite the paper as follows:

```
@misc{xiao2024flowbenchrevisitingbenchmarkingworkflowguided,
      title={FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents}, 
      author={Ruixuan Xiao and Wentao Ma and Ke Wang and Yuchuan Wu and Junbo Zhao and Haobo Wang and Fei Huang and Yongbin Li},
      year={2024},
      eprint={2406.14884},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.14884}, 
}
```
