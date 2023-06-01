import json
import os
import pandas as pd
import pyarrow as pa
import random
import gc

from tqdm import tqdm
from glob import glob
from collections import defaultdict

def path2rest(path, iid2pos, iid2neg, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    pos = iid2pos[name]
    neg = iid2neg[name]
    split = iid2split[name]
    return [binary, pos, neg, name, split]


def make_arrow(root, dataset_root, image_dataset=None):
    max_length = 0
    max_length = 0
    if image_dataset == None:
        image_dataset = dataset_root
    for split in ["val", "test", "train"]:
        iid2captions = defaultdict(list)
        iid2negtxts = defaultdict(list)
        iid2split = dict()
        with open(f"{root}/{split}/simple_conversations.json", "r") as fp:
            content = json.load(fp)
            for dialog in tqdm(content):
                conversation = dialog["conversation"]
                cur_context = []
                this_images = set()
                this_negtxts = []
                for idx, turn in enumerate(conversation):
                    turn = turn["turn"]
                    text = turn[0]['__TEXT__']
                    cur_context.append(text)
                    if len(turn)==1:
                        neg_txt = " ".join(cur_context[-3:])
                        this_negtxts.append(neg_txt)
                    if len(turn)>=2:
                        for k, value in enumerate(turn[1:]):
                            image = f"{value['__MEDIA__']}.jpg"
                            caps = " ".join(cur_context[-3:])
                            iid2captions[image].append(caps)
                            iid2split[image] = split
                            max_length = max(max_length, len(caps.split()))
                            this_images.add(image)
                for img in this_images:
                    iid2negtxts[img].append(this_negtxts)
            print("="*20," max_length : ", max_length,"="*20)  

        paths = list(glob(f"{image_dataset}/{split}/*.jpg"))
        random.shuffle(paths)
        caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
        if len(paths) == len(caption_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(caption_paths), len(iid2captions),
        )

        trunc = 2000000

        sub_len = int(len(caption_paths) // trunc)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * trunc : (sub + 1) * trunc]
            batches = [path2rest(path, iid2captions, iid2negtxts, iid2split) for path in tqdm(sub_paths)]

            print(f"{split} : ", len(batches))

            dataframe = pd.DataFrame(
                batches, columns=["image", "pos", "neg", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/mmdial_intent_{split}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del batches
            gc.collect() 

if __name__ == "__main__":
    root = "/data/pretrain/mmdial/MMDialogConversations"
    dataset_root = "/data/dataset"
    image_dataset = "/data/pretrain/mmdial/MMDialogImage"
    make_arrow(root, dataset_root, image_dataset)