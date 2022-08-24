# STAR3 Meet SQLova

## 数据预处理

2022 WAIC 黑客松蚂蚁财富赛道：AntSQL大规模金融语义解析中文Text-to-SQL挑战赛，从天池官网的比赛入口进入，https://tianchi.aliyun.com/competition/entrance/532009/introduction?spm=5176.12281949.0.0.72352448BL6ouP。

我们已经将这些文件处理成了模型可以训练的数据结构，文件夹中包含：`table.json, train_tok.json, dev_tok.json, testa_tok.json`。

后续会upload数据预处理的代码。现在可以直接使用处理好的数据。

<br>

## 安装ModelScope

```
# 不需要conda的可以忽略
conda create -n modelscope python=3.7
conda activate modelscope

# 直接运行安装
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

<br>

## 下载预处理好数据

就在放在git的根目录就行

```
# 训练数据 + A榜数据集
wget http://binhua-poc.oss-cn-beijing.aliyuncs.com/sqlova_data.zip -O sqlova_data.zip

# 训练数据 + A榜数据集 + B榜数据集
# wget http://binhua-poc.oss-cn-beijing.aliyuncs.com/sqlova_data_testb.zip -O sqlova_data.zip
# 让模型改为B榜数据集，需要修改train.py, 约797行, 将 mode = 'testa'，改为 mode = 'testb'

unzip sqlova_data.zip

```

<br>

## 模型训练

运行脚本：

```
python train.py \
    --do_train \
    --bS 16 \
    --num_target_layers 12 \
    --data_dir /your_git_path/sqlova_data/ \
    --output_dir /your_git_path/ant_tableqa/ \
    --output_name train_dev.log \
    --run_name sqlova-v1 \
    --bert_path /your_git_path/star3_tiny_model/damo/nlp_convai_text2sql_pretrain_cn/
```

这里`data_dir`指的是数据文件夹目录，`output_dir`指的是模型和log文件保存的路径，`output_name`是输出log的文件名，`run_name`是实验子文件夹名称，`bert_path`是预训练模型的文件夹目录。运行代码后开始训练过程。其他参数在`train.py`代码中的arguments中说明。

<br>

## 模型预测

运行脚本：

```
python train.py \
    --do_infer \
    --bS 1 \
    --num_target_layers 12 \
    --test_epoch 1 \
    --data_dir /your_git_path/sqlova_data/ \
    --output_dir /your_git_path/ant_tableqa/ \
    --output_name test.log \
    --run_name sqlova-v1 \
    --bert_path /your_git_path/star3_tiny_model/damo/nlp_convai_text2sql_pretrain_cn/
```

这里`data_dir`指的是数据文件夹目录，`output_dir`指的是模型和log文件保存的路径，`output_name`是输出log的文件名，`run_name`是实验子文件夹名称，需要和训练时候保持一致，`bert_path`是预训练模型的文件夹目录，`test_epoch`指的是需要测试的对应迭代轮数的模型。运行代码后开始预测。其他参数在`train.py`代码中的arguments中说明。预测结果会输出到路径`/your_git_path/ant_tableqa/sqlova-v1/final_test.jsonl`，该文件可以直接上传到比赛中获取结果。
