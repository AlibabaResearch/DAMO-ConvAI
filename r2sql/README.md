# R²SQL
The PyTorch implementation of paper [Dynamic Hybrid Relation Network for Cross-Domain Context-Dependent Semantic Parsing.](https://arxiv.org/pdf/2101.01686) (AAAI 2021)


## Requirements
The model is tested in python 3.6 with following requirements:
```
torch==1.0.0
transformers==2.10.0
sqlparse
pymysql
progressbar
nltk
numpy
six
spacy
```
All experiments on SParC and CoSQL datasets were run on NVIDIA V100 GPU with 32GB GPU memory.
* Tips: The 16GB GPU memory may appear out-of-memory error.

## Setup

The SParC and CoSQL experiments in two different folders, you need to download different datasets from [[SParC](https://yale-lily.github.io/spider) | [CoSQL](https://yale-lily.github.io/cosql)] to the `{sparc|cosql}/data` folder separately.
Another related data file could be download from [EditSQL](https://github.com/ryanzhumich/editsql/tree/master/data).
Then, download the database sqlite files from [[here](https://drive.google.com/file/d/1a828mkHcgyQCBgVla0jGxKJ58aV8RsYK/view?usp=sharing)] as `data/database`.

Download Pretrained BERT model from [[here](https://drive.google.com/file/d/1f_LEWVgrtZLRuoiExJa5fNzTS8-WcAX9/view?usp=sharing)] as `model/bert/data/annotated_wikisql_and_PyTorch_bert_param/pytorch_model_uncased_L-12_H-768_A-12.bin`.

Download Glove embeddings file (`glove.840B.300d.txt`) and change the `GLOVE_PATH` for your own path in all scripts. 

Download Reranker models from [[SParC reranker](https://drive.google.com/file/d/1cA106xgSx6KeonOxD2sZ06Eolptxt_OG/view?usp=sharing) | [CoSQL reranker](https://drive.google.com/file/d/1UURYw15T6zORcYRTvP51MYkzaxNmvRIU/view?usp=sharing)] as `submit_models/reranker_roberta.pt`, besides the roberta-base model could download from [here](https://drive.google.com/file/d/1LkTe-Z0AFg2dAAWgUKuCLEhSmtW-CWXh/view?usp=sharing) for `./[sparc|cosql]/local_param/`.

## Usage

Train the model from scratch.
```bash
./sparc_train.sh
```

Test the model for the concrete checkpoint:
```bash
./sparc_test.sh
```
then the dev prediction file will be appeared in `results` folder, named like `save_%d_predictions.json`.

Get the evaluation result from the prediction file:
```bash
./sparc_evaluate.sh
```
the final result will be appeared in `results` folder, named `*.eval`.

Similarly, the CoSQL experiments could be reproduced in same way.

---

You could download our trained checkpoint and results in here:

* SParC: [[log](https://drive.google.com/file/d/19ySQ_4x3R-T0cML2uJQBaYI2EyTlPr1G/view?usp=sharing) | [results](https://drive.google.com/file/d/12-kTEnNJKKblPDx5UIz5W0lVvf_sWpyS/view?usp=sharing)]
* CoSQL: [[log](https://drive.google.com/file/d/1QaxM8AUu3cQUXIZvCgoqW115tZCcEppl/view?usp=sharing) | [results](https://drive.google.com/file/d/1fCTRagV46gvEKU5XPje0Um69rMkEAztU/view?usp=sharing)]

### Reranker
If your want train your own reranker model, you could download the training file from here:

* SParC: [[reranker training data](https://drive.google.com/file/d/1XEiYUmDsVGouCO6NZS1yyMkUDxvWgCZ9/view?usp=sharing)]
* CoSQL: [[reranker training data](https://drive.google.com/file/d/1mzjywnMiABOTHYC9BWOoUOn4HnokcX8i/view?usp=sharing)]

Then you could train, test and predict it:

train:
```bash
python -m reranker.main --train --batch_size 64 --epoches 50
```

test:
```bash
python -m reranker.main --test --batch_size 64
```

predict:
```bash
python -m reranker.predict
```


## Improvements
We have improved the origin version (descripted in paper) and got more performance improvements :partying_face:!

Compare with the origin version, we have made the following improvements：

* add the self-ensemble strategy for prediction, which use different epoch checkpoint to get final result. In order to easily perform this strategy, we remove the task-related representation in Reranker module.
* remove the decay function in DCRI, we find that DCRI is unstable with decay function, so we let DCRI degenerate into vanilla cross attention.
* replace the BERT-based with RoBERTa-based model for Reranker module.

The final performance comparison on dev as follows:

<table>
  <tr>
    <th></th>
    <th colspan="2">SParC</th>
    <th colspan="2">CoSQL</th>
  </tr>
  <tr>
    <td></td>
    <td>QM</td>
    <td>IM</td>
    <td>QM</td>
    <td>IM</td>
  </tr>
  <tr>
    <td>EditSQL</td>
    <td>47.2</td>
    <td>29.5</td>
    <td>39.9</td>
    <td>12.3</td>
  </tr>
  <tr>
    <td>R²SQL v1 (origin paper)</td>
    <td>54.1</td>
    <td>35.2</td>
    <td>45.7</td>
    <td>19.5</td>
  </tr>
  <tr>
    <td>R²SQL v2 (this repo)</td>
    <td>54.0</td>
    <td>35.2</td>
    <td>46.3</td>
    <td>19.5</td>
  </tr>
  <tr>
    <td>R²SQL v2 + ensemble </td>
    <td>55.1</td>
    <td>36.8</td>
    <td>47.3</td>
    <td>20.9</td>
  </tr>
</table>

## Citation
Please star this repo and cite paper if you want to use it in your work.


## Acknowledgments
This implementation is based on ["Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions"](https://github.com/ryanzhumich/editsql) EMNLP 2019.

