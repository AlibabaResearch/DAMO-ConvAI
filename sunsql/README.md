# SUN: Exploring Intrinsic Uncertainties in Text-to-SQL Parsers

This repository contains code for the COLING 2022 paper [SUN: Exploring Intrinsic Uncertainties in Text-to-SQL Parsers].

If you use SUN in your work, please cite it as follows:

```
@inproceedings{qin2022sun,
    title={SUN: Exploring Intrinsic Uncertainties in Text-to-SQL Parsers},
    author={Bowen Qin, Lihan Wang, Binyuan Hui, Bowen Li, Pengxiang Wei, Binhua Li, Fei Huang, Luo Si, Min Yang, Yongbin Li},
    booktitle={COLING},
    year={2022}
}
```

## Codebase

### Prepare Environment

The setup of the environment is exactly the same as that of [LGESQL](https://github.com/rhythmcao/text2sql-lgesql):

The environment-related commands are provided in `setup.sh`.

```bash
sh setup.sh
```

### Download dataset.

Download, unzip and rename the [spider_sun.zip](https://drive.google.com/file/d/1KfBmF3iJ2HPRAOV6YBb-Vz0y64_86tej/view) into the directory `data`. In which, train_spider.json contains all pairs of data and train_rd.json contains the others data.

### Preprocess dataset.

Preprocess the train and dev dataset, including input normalization, schema linking, graph construction and output actions generation.

```bash
./run/run_preprocessing.sh
```

### Training

Training SUN with:

```
./run/run_lgesql_plm.sh msde electra-large-discriminator
```

### Evaluation

For evaluation, see `run/run_evaluation.sh` and `run/run_submission.sh` (eval from scratch) for reference.


## Acknowledgements

This implementation is based on [RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers.](https://github.com/microsoft/rat-sql) and [LGESQL: Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations](https://github.com/rhythmcao/text2sql-lgesql). Thanks to the author for releasing the code.
