<p align="center">
    <img src=assets/mmevol.jpg  width="45%"> <br>
    <!-- <span><b>Empowering Multimodal Large Language Models with Evol-Instruct</b></span> -->
</p>

# MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct

This is the official data collection of the paper "MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct", the dataset and checkpoint will be released soon.

We are continuously refactoring our code, be patient and wait for the latest updates!

## 🔗 Links
- Project Web: https://mmevol.github.io/

- Arxiv Paper: https://arxiv.org/pdf/2409.05840

- Code: Coming soon

## 🧪 Dataset Details

The Tongyi-ConvAI generates this dataset for multi-modal supervised fine-tuning. This dataset was used to train **Evol-Llama3-8B-Instruct** and **Evol-Qwen2-7B** reported in [our paper](https://arxiv.org/pdf/2409.05840).

To create this dataset, we first selected 163K Seed Instruction Tuning Dataset for Evol-Instruct, then we enhance data quality through an iterative process that involves a refined combination of fine-grained perception, cognitive reasoning, and interaction evolution. This process results in the generation of a more complex and diverse image-text instruction dataset, which in turn empowers MLLMs with enhanced capabilities.

Below we showcase the detailed data distribution of the SEED-163K, which is prepared for multi-round evolution mentioned above:

<p align="center">
    <img src=assets/seed_dis.jpg  width="95%"> <br>
    <span><b>Fig. 2. SEED-163K: 163K Curated Seed Instruction Tuning Dataset for Evol-Instruct</b></span>
</p>



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
