import json
import os
import pandas as pd
import pyarrow as pa
import random
import gc

from tqdm import tqdm
from glob import glob
from collections import defaultdict

def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    max_length = 0
    # words = 40
    json_list = list(glob(f"{root}/jsonfile/*/*.json"))
    dialogs = list()
    for js in json_list:
        split = js.split('/')[-2]
        with open(f"{js}","r") as fp:
            dialog = json.load(fp)
            for dial in dialog:
                dial["split"] = split
            dialogs += dialog

    iid2captions = defaultdict(list)
    iid2messages = defaultdict(list)
    iid2split = dict()

    for dial in tqdm(dialogs):
        filename = dial["photo_id"].split("/")[-1]+".jpg"
        split = dial["split"]
        dial["share_id"] = 0
        iid2split[filename] = split
        for ctx in dial["dialogue"]:
            if ctx["user_id"] == dial["share_id"]: 
                if ctx["share_photo"] == True:
                    ctx_caption = " ".join(iid2messages[filename][-2:])
                    # ctx_caption = " ".join(ctx_caption.split()[-words:]) # 姑且设为40
                    iid2captions[filename].append(ctx_caption)
                    iid2messages[filename] = []
                    max_length = max(max_length, len(ctx_caption.split()))
                    break
                
                iid2messages[filename].append(ctx["message"])
    print("==================== max_length : ", max_length,"="*20)
    del iid2messages
    paths = list(glob(f"{root}/*/*.jpg"))
    random.shuffle(paths)

    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]
    del dialogs

    for split in ["train", "validation", "test"]:
        batches = [b for b in bs if b[-1] == split]
        print(f"{split} : ",len(batches))
        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/photochat_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del batches
        gc.collect()        