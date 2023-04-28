import json

import datasets
import os
from PIL import Image

_CITATION = """\
@article{talmor2021multimodalqa,
  title={MultiModalQA: Complex Question Answering over Text, Tables and Images},
  author={Talmor, Alon and Yoran, Ori and Catav, Amnon and Lahav, Dan and Wang, Yizhong and Asai, Akari and Ilharco, Gabriel and Hajishirzi, Hannaneh and Berant, Jonathan},
  journal={arXiv preprint arXiv:2104.06039},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the MMQA.
"""

_HOMEPAGE = "https://github.com/allenai/multimodalqa"

_LICENSE = "MIT License"

_URL = "https://github.com/allenai/multimodalqa/raw/master/dataset/"
_TRAINING_FILE = "MMQA_train.jsonl.gz"
_DEV_FILE = "MMQA_dev.jsonl.gz"
_TEST_FILE = "MMQA_test.jsonl.gz"
_TEXTS_FILE = "MMQA_texts.jsonl.gz"
_TABLES_FILE = "MMQA_tables.jsonl.gz"
_PASSAGE_FILE = "MMQA_texts.jsonl.gz"
_IMAGES_INFO_FILE = "MMQA_images.jsonl.gz"
_IMAGE_URL = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"
_IMAGES_FILE = "final_dataset_images"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "test": f"{_URL}{_TEST_FILE}",
    "texts": f"{_URL}{_TEXTS_FILE}",
    "tables": f"{_URL}{_TABLES_FILE}",
    "images_info": f"{_URL}{_IMAGES_INFO_FILE}",
    "passages": f"{_URL}{_PASSAGE_FILE}",
    "images": f"{_IMAGE_URL}"
}


class Images(object):
    def __init__(self, images_info_path, pictures_path):
        self.images_info_path = images_info_path
        self.pictures_path = pictures_path

        self.images_info_dict = {}
        self.images_pic_dict = {}

        with open(images_info_path, "r") as f:
            images_info = [json.loads(_line) for _line in f.readlines()]
        for image_info in images_info:
            # {
            #   "title": "Nintendo Entertainment System",
            #   "url": "https://en.wikipedia.org/wiki/Nintendo_Entertainment_System",
            #   "id": "991237d4689fa65507e9528a1fab7de3",
            #   "path": "991237d4689fa65507e9528a1fab7de3.jpg"
            # }
            self.images_info_dict[image_info['id']] = image_info

    def load_image(self, pic_id, open_by_pillow=False):
        picture_absolute_path = os.path.join(self.pictures_path, self.images_info_dict[pic_id]['path'])
        if open_by_pillow:
            return Image.open(picture_absolute_path)
        else:
            return picture_absolute_path

    def __getitem__(self, id):
        return {"pic": self.load_image(id), **self.images_info_dict[id]}


class MMQA(datasets.GeneratorBasedBuilder):
    """The MMQA dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table": datasets.features.Sequence(
                        {
                            "table_id": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "caption": datasets.Value("string"),
                            "header": datasets.features.Sequence(datasets.Value("string")),
                            "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                            "rows_with_links": datasets.features.Sequence(datasets.features.Sequence(
                                datasets.features.Sequence(datasets.features.Sequence(
                                    datasets.Value("string")))))
                        }
                    ),
                    "images": datasets.features.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "url": datasets.Value("string"),
                            "path": datasets.Value("string"),
                            "pic": datasets.Value("string")
                        }
                    ),
                    "passages": datasets.features.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "url": datasets.Value("string"),
                            "text": datasets.Value("string")
                        }
                    ),
                    "answer_text": datasets.Value("string"),
                    "supporting_context": datasets.features.Sequence(
                        {
                            "doc_id": datasets.Value("string"),
                            "doc_part": datasets.Value("string")
                        }
                    ),
                    "type": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "file_path": downloaded_files["train"],
                    "table_path": downloaded_files["tables"],
                    "images_path": os.path.join(downloaded_files["images"], _IMAGES_FILE),
                    "images_info_path": downloaded_files["images_info"],
                    "passage_path": downloaded_files["passages"]
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "file_path": downloaded_files["dev"],
                    "table_path": downloaded_files["tables"],
                    "images_path": os.path.join(downloaded_files["images"], _IMAGES_FILE),
                    "images_info_path": downloaded_files["images_info"],
                    "passage_path": downloaded_files["passages"]
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "file_path": downloaded_files["test"],
                    "table_path": downloaded_files["tables"],
                    "images_path": os.path.join(downloaded_files["images"], _IMAGES_FILE),
                    "images_info_path": downloaded_files["images_info"],
                    "passage_path": downloaded_files["passages"]
                }),
        ]

    def _generate_examples(self, split, file_path, table_path, images_path, images_info_path, passage_path):
        """Yields examples."""
        tables = {}
        with open(table_path, 'r') as f:
            for line in f:
                table = json.loads(line)
                tables[table["id"]] = table
        texts = {}
        with open(passage_path, 'r') as f:
            for line in f:
                text = json.loads(line)
                texts[text["id"]] = text

        mmqa_images = Images(images_info_path, images_path)

        with open(file_path, 'r') as f:
            count = 0
            for idx, line in enumerate(f):
                example = json.loads(line)
                count += 1
                example_table = []
                example_images = []
                example_texts = []

                # load table
                table_id = example['metadata']['table_id']
                rows_with_links = []
                for row in tables[table_id]["table"]["table_rows"]:
                    rows_with_links.append([])
                    for cell in row:
                        text, title, url = [], [], []
                        for link in cell['links']:
                            text.append(link['text'])
                            title.append(link['wiki_title'])
                            url.append(link['url'])
                        rows_with_links[-1].append([text, title, url])

                example_table.append({
                    "table_id": table_id,
                    "title": tables[table_id]["title"],
                    "caption": tables[table_id]["table"]["table_name"],
                    "header": [column["column_name"] for column in tables[table_id]["table"]["header"]],
                    "rows": [[cell["text"] for cell in row] for row in tables[table_id]["table"]["table_rows"]],
                    "rows_with_links": rows_with_links
                })

                # load image_docs
                for image_doc_id in example['metadata']['image_doc_ids']:
                    example_images.append(mmqa_images[image_doc_id])

                # load text_docs
                for text_doc_id in example['metadata']['text_doc_ids']:
                    # {
                    #   "title": "Hillaryland",
                    #   "url": "https://en.wikipedia.org/wiki/Hillaryland",
                    #   "id": "a7d9e6350bafc46b700e4d0739a39594",
                    #   "text": "Hillaryland was the self-designated name of a group of core advisors to Hillary Clinton, when she was First Lady of the United States and again when, as United States Senator, she was one of the Democratic Party candidates for President in the 2008 U.S. election."
                    # }
                    example_texts.append(texts[text_doc_id])

                if split in ['train', 'validation']:
                    yield count, {
                        "id": example["qid"],
                        "question": example["question"],
                        "table": example_table,
                        "images": example_images,
                        "passages": example_texts,
                        "answer_text": " | ".join([str(answer["answer"]) for answer in example["answers"]]),
                        "supporting_context": example['supporting_context'],
                        "type": example['metadata']['type']
                        #    supporting_context be like
                        #     {
                        #         "doc_id": "dcd7cb8f23737c6f38519c3770a6606f",
                        #         "doc_part": "table"
                        #     }
                    }
                else:
                    yield count, {
                        "id": example["qid"],
                        "question": example["question"],
                        "table": example_table,
                        "images": example_images,
                        "passages": example_texts,
                        "answer_text": "",
                        "supporting_context": [],
                        "type": ""
                    }
