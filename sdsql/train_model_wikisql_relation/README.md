
# Schema dependency Guided Language-to-SQL Network for WikiSQL

## Preprocessing
对数据进行预处理，包括schema link 和 struct link
knowledge_link.py -- schema link
struct_link.py -- 结构解析



## Data

把训练数据放到`data_and_model`目录下:
* dev_struct_tok.jsonl
* test_struct_tok.jsonl
* train_struct_tok.jsonl

## Train
```
python train.py
```

## Test
```
python predict.py
```
预测结果保存至 `sxron.json`

```
python regual_merge.py
```
merge 两种子任务的结果

## Result

### Overall
| Method | dev_lf | dev_ex | test_lf | test_ex |
| --- | --- | --- | --- | --- |
| IESQL | 87.9 | 92.6 | 87.8 | 92.5 |
| Ours | 86.0 | 91.9 | 85.6 | 91.6 |
| Ours2 |  | | 85.5 | 91.4 |
| Ours2 + merge | | | 86.2 | 91.9 |

### Fine grained
|  Method  | S_col | S_agg | W_no | W_col | W_op | W_val |
|  ----  | ----  | --- | --- | --- | --- | --- |
| IESQL  | 97.6 | 94.7 | 98.3 | 97.9 | 98.5 | 98.3 |  
| Ours  | 97.3 | 90.6 | 98.6 | 97.8 | 97.3 | 97.9 |
| Ours2 | 97.2 | 90.8 | 97.2 | 97.5 | 97.2 | 97.5 |
| Ours2 + merge | 97.3 | 90.9 | 98.8 | 98.1 | 97.7 | 98.3 |
