import json
import os

import numpy as np
import glob
import tensorflow.compat.v1 as tf
from collections import defaultdict

from tqdm import tqdm


class DataParser():
    def __init__(self,
                 filepath="aburns4/WikiWeb2M/wikiweb2m-*",
                 path=""):
        self.filepath = filepath
        self.path = path
        self.data = defaultdict(list)

    def parse_data(self):
        context_feature_description = {
            'split': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
            'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_content_sections_without_table_list': tf.io.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_feature_description = {
            'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_title': tf.io.VarLenFeature(dtype=tf.string),
            'section_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
            'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
            'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
            'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_sequence_example(example_proto,
                                                       context_feature_description,
                                                       sequence_feature_description)

        suffix = '.tfrecord*'

        data_path = glob.glob(self.path + self.filepath + suffix)
        # data_path = data_path[:1]
        print(data_path)
        raw_dataset = tf.data.TFRecordDataset(data_path, compression_type='GZIP')
        parsed_dataset = raw_dataset.map(_parse_function)

        import concurrent.futures
        from tqdm import tqdm

        def process_item(d):
            """独立处理单个数据项，返回 (split, instance)"""
            context = d[0]
            sequence_feature = d[1]

            split = context['split'].numpy().decode()
            page_title = context['page_title'].numpy().decode()
            page_url = context['page_url'].numpy().decode()
            clean_page_description = context['clean_page_description'].numpy().decode()
            raw_page_description = context['raw_page_description'].numpy().decode()
            is_page_description_sample = context['is_page_description_sample'].numpy()
            page_contains_images = context['page_contains_images'].numpy()
            page_content_sections_without_table_list = context['page_content_sections_without_table_list'].numpy()

            section_title = [s.decode('utf-8') for s in sequence_feature['section_title'].values.numpy()]
            section_text = [s.decode('utf-8') for s in sequence_feature['section_text'].values.numpy()]
            section_image_url = [s.decode('utf-8') for s in sequence_feature['section_image_url'].values.numpy()]
            section_image_captions = [s.decode('utf-8') for s in
                                      sequence_feature['section_image_captions'].values.numpy()]
            section_image_clean_ref_desc = [s.decode('utf-8') for s in
                                            sequence_feature['section_image_clean_ref_desc'].values.numpy()]
            section_contains_images = sequence_feature['section_contains_images'].values.numpy().tolist()

            instance = {
                "page_title": page_title,
                "page_url": page_url,
                "clean_page_description": clean_page_description,
                "page_contains_images": int(page_contains_images),
                "section_title": section_title,
                "section_text": section_text,
                "section_image_url": section_image_url,
                "section_image_captions": section_image_captions,
                "section_image_clean_ref_desc": section_image_clean_ref_desc,
                "section_contains_images": section_contains_images,
            }

            return split, instance

        from concurrent.futures import ThreadPoolExecutor

        all_data = {"train": [], "val": [], "test": []}

        def worker(d):
            return process_item(d)

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 直接 map，避免先遍历全部
            futures = []
            count = 0
            for d in parsed_dataset:
                if count >= 100000:
                    break
                futures.append(executor.submit(worker, d))
                count += 1

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                split, instance = future.result()
                all_data[split].append(instance)
        return all_data



parser = DataParser()
all_data = parser.parse_data()

# Save JSON
out_dir = "aburns4/WikiWeb2M"
os.makedirs(out_dir, exist_ok=True)
for s in ['train', 'val', 'test']:
    with open(f"{out_dir}/pretrain-{s}.json", 'w', encoding='utf-8') as f:
        json.dump(all_data[s], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_data[s])} samples to pretrain-{s}.json")
