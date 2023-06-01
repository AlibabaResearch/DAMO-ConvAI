import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob


def path2rest(path, iid2captions):
    split, name = path.split("/")[-2:]
    split = split.split("_")[-1]
    iid = name

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]

    return [
        binary,
        captions,
        iid,
        split,
    ]


def make_arrow(root, dataset_root, image_dataset=None):
    if image_dataset == None:
        image_dataset = dataset_root
    for split in ["train"]:#["test", "val", "train"]:
        with open(f"{root}/blip_captions_{split}.json", "r") as fp:
            captions = json.load(fp)

        iid2captions = dict()
        for cap in tqdm(captions):
            iid = cap["image_id"]+".jpg"
            iid2captions[iid] = [cap["caption"]]

        paths = list(glob(f"{image_dataset}/{split}/*"))
        random.shuffle(paths)
        caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
        if len(paths) == len(caption_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(caption_paths), len(iid2captions),
        )

        trunc = 2500000

        sub_len = int(len(caption_paths) // trunc)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * trunc : (sub + 1) * trunc]
            bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/mmdialog_caption_{split}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)

            del sub_paths
            del dataframe
            del table
            del bs
            gc.collect()


if __name__ == "__main__":
    root = "/data/pretrain/MMDialog/MMDialogCaption"
    dataset_root = "/data/dataset"
    image_dataset = "/data/pretrain/MMDialog/MMDialogImage"
    make_arrow(root, dataset_root, image_dataset)