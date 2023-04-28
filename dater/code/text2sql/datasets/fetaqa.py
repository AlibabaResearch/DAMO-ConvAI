import os
import datasets
from utils.wtq.utils import _load_table_w_page as _load_table
import json

class FetaQA(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = {
            "feta_id": datasets.Value("int32"),
            "table": {
                "id": datasets.Value("string"),
                "header": datasets.features.Sequence(datasets.Value("string")),
                "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                "table_page_title": datasets.Value("string"),
            },
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
        }

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'WikiTableQuestions-master')
        data_dir = 'dataset/'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.jsonl"),
                            "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.jsonl"),
                            "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.jsonl"),
                            "data_dir": data_dir},
            ),

        ]

    def _generate_examples(self, statements_file, all_csv_path, info_path):

            with open('fetaQA-v1_test.jsonl') as f:
                lines  = f.readlines()
                for i,l in feta_id(lines):
                    dic = json.loads(l)
                    feta_id = dic['feta_id']
                    caption = dic['table_page_title']
                    question = dic['question']
                    answer = dic["answer"]
                    header = dic['table_array'][0]
                    rows = dic['table_array'][1:]
                    yield f"{i}", {
                    "id": feta_id,
                    "table": {
                        "id": feta_id,
                        "header": header,
                        "rows": rows,
                        "caption": caption
                    },
                    "question": question,
                    "answer": answer
                }
