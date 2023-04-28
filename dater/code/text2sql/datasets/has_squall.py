# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Squall: On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries"""


import json
import os
import datasets
import shutil
from utils.wtq.utils import _load_table_w_page as _load_table

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{Shi:Zhao:Boyd-Graber:Daume-III:Lee-2020,
	Title = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
	Author = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
	Booktitle = {Findings of EMNLP},
	Year = {2020},
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/tzshi/squall"

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip"
_SQUALL_URL = "https://github.com/tzshi/squall/archive/refs/heads/main.zip"

from utils.wtq.utils import WTQDBEngine, process_table_structure, retrieve_wtq_query_answer

class WikiTableQuestion(datasets.GeneratorBasedBuilder):
    """The Squall dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {
                        "page_title": datasets.Value("string"),
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))
                    },
                    "sql": datasets.Value("string"),
                    "answer_text": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'WikiTableQuestions-master')
        squall_dir = os.path.join(dl_manager.download_and_extract(_SQUALL_URL), 'squall-main')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/random-split-1-train.tsv"),
                    "data_dir": data_dir,
                    "squall_path": os.path.join(squall_dir, "data/squall.json"),
                    "squall_tables_path": os.path.join(squall_dir, "tables/json"),
                    "squall_db_path": os.path.join(squall_dir, "tables/db"),
                    "squall_tmp_db_path": os.path.join(squall_dir, "tables/tmp_db"),

                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/random-split-1-dev.tsv"),
                    "data_dir": data_dir,
                    "squall_path": os.path.join(squall_dir, "data/squall.json"),
                    "squall_tables_path": os.path.join(squall_dir, "tables/json"),
                    "squall_db_path": os.path.join(squall_dir, "tables/db"),
                    "squall_tmp_db_path": os.path.join(squall_dir, "tables/tmp_db"),

                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/pristine-unseen-tables.tsv"),
                    "data_dir": data_dir,
                    "squall_path": os.path.join(squall_dir, "data/squall.json"),
                    "squall_tables_path": os.path.join(squall_dir, "tables/json"),
                    "squall_db_path": os.path.join(squall_dir, "tables/db"),
                    "squall_tmp_db_path": os.path.join(squall_dir, "tables/tmp_db"),

                },
            ),

        ]

    def _generate_examples(self, filepath, data_dir, squall_path, squall_tables_path, squall_db_path,
                           squall_tmp_db_path):
        if not os.path.exists(squall_tmp_db_path):
            os.makedirs(squall_tmp_db_path)

        # source table should not be truncated!
        src_table_content_map = {}
        # tgt table should be truncated!
        tgt_table_content_map = {}
        table_drop_rows_map = {}
        db_engine_map = {}

        for table_json_file in os.listdir(squall_tables_path):
            table_id = table_json_file[:-5]
            check_table_file = open(os.path.join(squall_tables_path, table_json_file), "r", encoding="utf8")
            src_table_content = json.load(check_table_file)
            src_table_content = process_table_structure(src_table_content)
            src_table_content_map[table_id] = json.loads(json.dumps(src_table_content))
            tgt_table_content_map[table_id] = src_table_content

        for table_db_file in os.listdir(squall_db_path):
            table_id = table_db_file[:-3]
            # copy table db file into a temp file since we may delete some rows
            database_path = os.path.join(squall_db_path, table_db_file)
            temp_database_path = os.path.join(squall_tmp_db_path, table_db_file)
            if os.path.exists(temp_database_path):
                os.remove(temp_database_path)
            # future operations on the temp db to avoid effecting the original database
            shutil.copy(database_path, temp_database_path)
            db_engine_map[table_id] = WTQDBEngine(temp_database_path)
            if table_id in table_drop_rows_map and len(table_drop_rows_map[table_id]) != 0:
                table_drop_rows = table_drop_rows_map[table_id]
                db_engine_map[table_id].delete_rows(table_drop_rows)

        """Yields examples."""
        squall_id_map = {}
        with open(squall_path) as f:
            squall_data = json.load(f)
            for squall_item in squall_data:
                squall_id_map[squall_item["nt"]] = squall_item

        # data_id, question, table_id, gold_result_str
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # skip the header
                if idx == 0:
                    continue
                data_id, question, table_id, gold_result_str = line.strip("\n").split("\t")

                if data_id in squall_id_map.keys():
                    # Data annotation from WikiTableQuestion dataset
                    table = _load_table(os.path.join(data_dir, table_id.replace('.csv', '.tsv')))
                    gold_result = gold_result_str.split('|')

                    # Data annotation from Squall dataset.
                    squall_data_item = squall_id_map[data_id]
                    squall_table_id = squall_data_item["tbl"]
                    sql_struct = squall_data_item["sql"]
                    engine, src_table_content = db_engine_map[squall_table_id], src_table_content_map[squall_table_id]
                    try:
                        encode_sql_str, _, exec_sql_str = retrieve_wtq_query_answer(engine, table, sql_struct)
                    except IndexError as e:
                        # In case header is modified.
                        encode_sql_str, _, exec_sql_str = retrieve_wtq_query_answer(engine, src_table_content, sql_struct)

                    yield idx, {
                        "id": data_id,
                        "question": question,
                        "table_id": table_id,
                        "table": table,
                        "sql": encode_sql_str,
                        "answer_text": gold_result,
                    }
                else:
                    continue
