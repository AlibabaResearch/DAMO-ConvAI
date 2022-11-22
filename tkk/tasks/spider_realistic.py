# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Spider-Realisitc: Spider Realistic Dataset for evaluating Text-SQL models"""

import json
import os
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import datasets
from typing import List, Generator, Any, Dict, Tuple

logger = datasets.logging.get_logger(__name__)
data_path = f"{os.getcwd()}/data/robustness/spider_realistic/"

_CITATION = """
@article{deng2020structure,
  title={Structure-Grounded Pretraining for Text-to-SQL},
  author={Deng, Xiang and Awadallah, Ahmed Hassan and Meek, Christopher and Polozov, Oleksandr and Sun, Huan and Richardson, Matthew},
  journal={arXiv preprint arXiv:2010.12773},
  year={2020}
}
"""

_DESCRIPTION = """\
    Spider-realistic: Original spider dev data modified to remove explicit mention of column names from utterance for better evaluation.
"""

_HOMEPAGE = "https://zenodo.org/record/5205322#.Yh-B1uhByUl"

_LICENCE = "CC BY-SA 4.0"

_URL = "https://drive.google.com/uc?export=download&confirm=9iBg&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class SpiderRealistic(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [datasets.BuilderConfig(
        name = "spider-realistic",
        version = VERSION,
        description="Spider-realistic: Original spider dev data modified to remove explicit mention of column names from utterance for better evaluation."
    )]


    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                'question': datasets.Value('string'),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id":datasets.Value("int32"),
                        "column_name": datasets.Value("string")
                    }
                ),
                "db_column_types":datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({
                    "column_id":datasets.Value("int32"),
                }),
                "db_foreign_keys":datasets.features.Sequence(
                    {
                        "column_id":datasets.Value("int32"),
                        "other_column_id":datasets.Value("int32")
                    }
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name = datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": data_path + "spider-realistic.json",
                    "db_path": downloaded_filepath + "/spider/database",
                },
            ),
        ]
    
    def _generate_examples(
        self, data_filepath: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    
        logger.info("generating examples form = %s", data_filepath)
        with open(data_filepath, encoding='utf-8') as f:
            spider_realistic = json.load(f)
            for idx, sample in enumerate(spider_realistic):
                db_id = sample['db_id']
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]
                yield idx, {
                    "query": sample['query'],
                    "question": sample['question'],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                }
