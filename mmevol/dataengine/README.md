# Data construction pipeline for MMEvol-480k.

<p align="center">
    <img src="assets/mmevol_logo.png" width="50%" height="50%">
</p>

<div align="center">
<br>
<a href="https://scholar.google.com/citations?user=phg8yxoAAAAJ&hl=zh-CN&oi=ao">Run Luo</a><sup><span>1,2*</span></sup>, 
<a>Haonan Zhang</a><sup><span>3*</span></sup>,
<a>Longze Chen</a><sup><span>1,2*</span></sup>,
<a>Ting-En Lin</a><sup><span>3*</span></sup>,
<a>Xiong Liu</a><sup><span>3</span></sup>,
<a>Yuchuan Wu</a><sup><span>3</span></sup>,
<a>Min Yang</a><sup><span>1,2🌟</span></sup>,
<a>Yongbin Li</a><sup><span>3🌟</span></sup>,
<br>
<a>Minzheng Wang<sup><span>2</span></sup>,
<a>Pengpeng Zeng<sup><span>4</span></sup>,
<a>Lianli Gao<sup><span>5</span></sup>,
<a>Heng Tao Shen<sup><span>4</span></sup>,
<a>Yunshui Li<sup><span>1,2</span></sup>,
<a>Xiaobo Xia<sup><span>6</span></sup>,
<a>FeiHuang<sup><span>3</span></sup>,
<a>Jingkuan Song<sup><span>4🌟</span></sup>,
<br>

\* Equal contribution 🌟 Corresponding author

<sup>1</sup> Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences<br>
<sup>2</sup> University of Chinese Academy of Sciences<br>
<sup>3</sup> Alibaba Group
<sup>4</sup> Tongji University 
<sup>5</sup> Independent Researcher
<sup>6</sup> The University of Sydney<br>
    
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) <a href='https://arxiv.org/pdf/2409.05840'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/models/Tongyi-ConvAI/MMEvol'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a> <a href='https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'> <a href='https://mmevol.github.io/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Project-Page-green'></a></a>

</div>


<font size=5><div align='center' >  [[📖 arXiv Paper](https://arxiv.org/pdf/2409.05840)] [[📊 Dataset](https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol)] [[🏆 Models](https://huggingface.co/models/Tongyi-ConvAI/MMEvol)]  </div></font>

Follow the instructions below to generate MMEvol-480k.

1. Download SEED-163k json file (`mm_seed_no_evo_163k.json`) from [🤗 huggingface](https://huggingface.co/datasets/Tongyi-ConvAI/MMEvol/tree/main/jsons), and place it under the `./dataengine/datasets` path.
2. Execute preprocessing code under `dataengine/datasets` path to extract each sample to the `meta_data` folder by:
```python
python dataengine/datasets/process.py
```
3. Prepare the data storage folder by referring to the format of `./dataengine/evolution/folder_template`, you can just copy folder_template and name it as your data name as you like, _e.g._, mmevol_1k_evo.json.
4. Ensure that your `api_base` and `key` are correctly configured before starting generation. You should put your key and api_base on both:

- lines 129-130 in dataengine/multi_round.py
- lines 126-127 in dataengine/score_process/difficulty_scoring_v123.py
5. Run the following code to begin the three-round data evolution: 
```python
python dataengine/multi_round.py
```
Three rounds of evolution will be performed based on the SEED-163k, and data filtering will be performed at the end of each round of evolution. The final evolution data will be stored under `./datasets` paths

**License**: Please follow [Meta Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and [Gemma License](https://www.kaggle.com/models/google/gemma/license/).

## 📚 Citation

```bibtex
@article{luo2024mmevol,
  title={Mmevol: Empowering multimodal large language models with evol-instruct},
  author={Luo, Run and Zhang, Haonan and Chen, Longze and Lin, Ting-En and Liu, Xiong and Wu, Yuchuan and Yang, Min and Wang, Minzheng and Zeng, Pengpeng and Gao, Lianli and others},
  journal={arXiv preprint arXiv:2409.05840},
  year={2024}
}
```

**Contact**:

- Run Luo — r.luo@siat.ac.cn

- Haonan Zhang — zchiowal@gmail.com
