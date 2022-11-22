# SParC: Cross-Domain Semantic Parsing in Context

SParC is a large dataset for complex, cross-domain, and context-dependent(multi-turn) semantic parsing and text-to-SQL task (interactive natural language interfaces for relational databases) built upon our [Spider task](https://yale-lily.github.io/spider). It is released along with our ACL 2019 paper: [SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285). This repo contains all code for evaluation and baselines used in our paper. Please refer to [the SParC task page](https://yale-lily.github.io/sparc) for more general introduction and the leaderboard.

### Changelog

- `6/20/2019` SParC 1.0 is released! Please report any annotation errors under [this issue](https://github.com/taoyds/sparc/issues/1), we really appreciate your help and will update the data release in this summer!

### Citation

The dataset is annotated by 14 college students. When you use the SParC dataset, we would appreciate it if you cite the following:

```
@InProceedings{Yu&al.19,
  title     = {SParC: Cross-Domain Semantic Parsing in Context},
  author    = {Tao Yu and Rui Zhang and Michihiro Yasunaga and Yi Chern Tan and Xi Victoria Lin and Suyi Li and Heyang Er, Irene Li and Bo Pang and Tao Chen and Emily Ji and Shreya Dixit and David Proctor and Sungrok Shim and Jonathan Kraft, Vincent Zhang and Caiming Xiong and Richard Socher and Dragomir Radev},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{Yu&al.18c,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev}
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  address   = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  year      = 2018
}
```

### Installation

`evaluation.py` and `process_sql.py` were tested in Python 2.7. 


### Data Content and Format

#### Question, SQL, and Parsed SQL

Each file in`train.json` and `dev.json` contains the following fields:
- `question`: the natural language question
- `question_toks`: the natural language question tokens
- `database_id`: the database id to which this interaction is addressed.
- `interaction`: the query interaction including multiple DB query questions. For each question in the interaction, it includes:
  - `utterance`: the natural language question
  - `utterance_toks`: the natural language question tokens
  - `query`: the SQL query corresponding to the question. 
  - `sql`: parsed results of this SQL query using `process_sql.py`. Please refer to the [Spider Github page](https://github.com/taoyds/spider) for the detailed documentation.
- `final`: the final interaction query goal
  - `utterance`: the natural language question of the final interaction goal
  - `query`: the SQL query corresponding to the final interaction goal.
  
```
{
        "database_id": "car_1", 
        "interaction": [
            {
                "query": "Select T1.id, T2.model from cars_data as T1 join car_names as T2 on T1.id = T2.MakeId where T1.year = '1970';", 
                "utterance_toks": [
                    "What", 
                    "are", 
                    "the", 
                    ...
                    "?"
                ], 
                "utterance": "What are the ids, and models of the cars were made in 1970?", 
                "sql": {
                    "orderBy": [], 
                    "from": {
                        "table_units": [
                            [
                                "table_unit", 
                                5
                            ...
                    }, 
                    "union": null, 
                    "except": null, 
                    
            }, 
            {
                "query": "Select T1.horsepower, T1.mpg, T1.id, T2.model from cars_data as T1 join car_names as T2 on T1.id = T2.MakeId where T1.year = '1970';", 
                "utterance_toks": [
                    "Show", 
                    ...
                    "well", 
                    "?"
                ], 
                "utterance": "Show their horsepower and MPG as well?", 
                "sql": {
                    "orderBy": [], 
                    "from": {
                        ...
                    ]
                }
            }, 
            {
                "query": "SELECT T1.Maker, T4.horsepower, T4.mpg, T4.id, T2.model FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  '1970';", 
                "utterance_toks": [
                    "Also", 
                    ...
                    "makers", 
                    "!"
                ], 
                "utterance": "Also provide the names of their makers!", 
                "sql": {
                    "orderBy": [], 
                    "from": {
                        ...
                }
            }, 
            {
                "query": "SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  '1970';", 
                "utterance_toks": [
                    "Just", 
                    "show", 
                    ...
                    "makers", 
                    "."
                ], 
                "utterance": "Just show a unique list of all these different makers.", 
                "sql": {
                    ...
                    "union": null, 
                    "except": null, 
                    "having": [], 
                    "limit": null, 
                    "intersect": null, 
                    "where": [
                        [
                            false, 
                           ...
            }
        ], 
        "final": {
            "query": "SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  '1970';", 
            "utterance": "Find the name of the makers that produced some cars in the year of 1970?"
        }, 
}
```

#### Tables

`tables.json` contains the following information for each database:
- `db_id`: database id
- `table_names_original`: original table names stored in the database.
- `table_names`: cleaned and normalized table names. We make sure the table names are meaningful. [to be changed]
- `column_names_original`: original column names stored in the database. Each column looks like: `[0, "id"]`. `0` is the index of table names in `table_names`, which is `city` in this case. `"id"` is the column name. 
- `column_names`: cleaned and normalized column names. We make sure the column names are meaningful. [to be changed]
- `column_types`: data type of each column
- `foreign_keys`: foreign keys in the database. `[3, 8]` means column indices in the `column_names`. These two columns are foreign keys of two different tables.
- `primary_keys`: primary keys in the database. Each number is the index of `column_names`.


```
{
    "column_names": [
      [
        0,
        "id"
      ],
      [
        0,
        "name"
      ],
      [
        0,
        "country code"
      ],
      [
        0,
        "district"
      ],
      .
      .
      .
    ],
    "column_names_original": [
      [
        0,
        "ID"
      ],
      [
        0,
        "Name"
      ],
      [
        0,
        "CountryCode"
      ],
      [
        0,
        "District"
      ],
      .
      .
      .
    ],
    "column_types": [
      "number",
      "text",
      "text",
      "text",
         .
         .
         .
    ],
    "db_id": "world_1",
    "foreign_keys": [
      [
        3,
        8
      ],
      [
        23,
        8
      ]
    ],
    "primary_keys": [
      1,
      8,
      23
    ],
    "table_names": [
      "city",
      "sqlite sequence",
      "country",
      "country language"
    ],
    "table_names_original": [
      "city",
      "sqlite_sequence",
      "country",
      "countrylanguage"
    ]
  }
```


#### Databases

All table contents are contained in corresponding SQLite3 database files.


### Data Process

Please refer to [the Spider Github page](https://github.com/taoyds/spider/tree/master/preprocess) for parsing SQL queries .

### Evaluation

We follow the Spider evaluation methods to compute Component Matching, Exact Set Matching, and Execution Accuracies. Check out more details at [the Spider Github page](https://github.com/taoyds/spider).

You can find some evaluation examples [here](https://github.com/taoyds/sparc/tree/master/evaluation_examples).

```
python evaluation.py --gold [gold file] --pred [predicted file] --etype [evaluation type] --db [database dir] --table [table file]

arguments:
  [gold file]        gold.txt file where each line is `a gold SQL \t db_id`, and interactions are seperated by one empty line.
  [predicted file]   predicted sql file where each line is a predicted SQL, and interactions are seperated by one empty line.
  [evaluation type]  "match" for exact set matching score, "exec" for execution score, and "all" for both.
  [database dir]     directory which contains sub-directories where each SQLite3 database is stored.
  [table file]       table.json file which includes foreign key info of each database.
  
```

### Baseline Models

Please go to [this Github page](https://github.com/ryanzhumich/sparc_atis_pytorch) for the Pytorch implementation of the CD-Seq2Seq baseline that can run on both SParC and ATIS tasks.

### FAQ



