# Workshop Baseline
## Preparation
1. Download the [dev.json](https://modelscope.cn/api/v1/datasets/DAMO_ConvAI/FrViDoc2Bot/repo?Revision=master&FilePath=dev.json) file to your working directory.
2. Install requirements.
```bash
pip install -r requirements.txt
```

## Reproduce
You can reproduce our results with the following command:
```bash
bash run.sh
```
Or you can run the steps you need.

## Training
### Retrieval
```bash
python train_retrieval.py
```
This produces the model **DAMO_ConvAI/nlp_convai_retrieval_pretrain/**


### Rerank
```bash
python train_rerank.py
```
This produces the model **output/**


### Generation
```bash
python train_generation.py
```
This produces the model **DAMO_ConvAI/nlp_convai_generation_pretrain/**

## Testing
### Retrieval
```bash
python inference_retrieval.py
```
This produces the input file **input.jsonl** and the retrieval result **DAMO_ConvAI/nlp_convai_retrieval_pretrain/evaluate_result.json**


### Rerank
```bash
python inference_rerank.py
```
This produces the rerank result **rerank_output.jsonl**

### Generation
```bash
python inference_generation.py
```
This produces the generation result **outputStandardFile.json**, you can submit the document directly.

## Baseline Result
**outputStandardFileBaseline.json** is the output of our baseline method.

| Split    |  F1   | Sacrebleu | Rouge-L | Score  |
|----------|:-----:|:---------:|:-------:|:------:|
| dev.json | 58.55 |   42.03   |  55.83  | 156.42 |

## Useful Resources
We also provid:

1. The original Vietnamese and French label data, please refer to [French and Vietnamese document-grounded dialogue data set](https://modelscope.cn/datasets/DAMO_ConvAI/FrViDoc2Bot/summary).
2. Our organized Chinese document-grounded dialogue data from Doc2Bot (Doc2Bot: Accessing Heterogeneous Documents via Conversational Bots, EMNLP 2022), please refer to [Chinese document-grounded dialogue data set](https://modelscope.cn/datasets/DAMO_ConvAI/ZhDoc2BotDialogue/summary)
3. Our organized English document-grounded dialogue data from Multidoc2dial (MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents, EMNLP 2021), please refer to [English document-grounded dialogue data set](https://modelscope.cn/datasets/DAMO_ConvAI/EnDoc2BotDialogue/summary)

We hope that participants can leverage the linguistic similarities, for example, a large number of Vietnamese words are derived from Chinese, and English and French both belong to the Indo-European language family, to improve their models' performance in Vietnamese and French.

