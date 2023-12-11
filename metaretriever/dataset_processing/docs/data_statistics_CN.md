# 数据统计脚本

``` bash
python scripts/data_statistics.py \
    -data converted_data/text2spotasoc/
    -f csv
```

- data: 目标文件夹，遍历文件夹下包含 record.schema 的子文件夹，跳过所有的命名中包含 shot 和 rario 的文件夹
- f: 输出的表格形式，常见中 simple（默认），latex，html
