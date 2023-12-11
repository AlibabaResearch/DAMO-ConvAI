# 低资源数据采样

详细脚本见 `run_sample.bash`, 自动生成所有数据


## 低数据比例采样

``` text
 $ python scripts/sample_data_ratio.py -h
usage: sample_data_ratio.py [-h] [-src SRC] [-tgt TGT] [-seed SEED]

optional arguments:
  -h, --help  show this help message and exit
  -src SRC
  -tgt TGT
  -seed SEED
```

样例：

``` bash
python scripts/sample_data_ratio.py \
  -src converted_data/text2spotasoc/entity/mrc_conll03 \
  -tgt test_conll03_ratio 
```

对所有数据文件夹的train.json取指定 0.01 0.05 0.1 比例的数据

## N-shot 数据采样

``` text
 $ python scripts/sample_data_shot.py -h
usage: sample_data_shot.py [-h] -src SRC -tgt TGT -task {entity,relation,event} [-seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -src SRC              Source Folder Name
  -tgt TGT              Target Folder Name, n shot sampled
  -task {entity,relation,event}
                        N-Shot Task name
  -seed SEED            Default is None, no random
```

样例：

``` bash
python scripts/sample_data_shot.py \
  -src converted_data/text2spotasoc/entity/mrc_conll03 \
  -tgt test_conll03_shot \
  -task entity
```

1. 读取数据文件夹的 `entity.schema`
2. 根据每个类别采样 1 5 10 个样例合成最终数据
