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

    iid2pos = defaultdict(list)
    # iid2messages = defaultdict(list)
    iid2neg = defaultdict(list)
    iid2split = dict()

    for dial in tqdm(dialogs):
        filename = dial["photo_id"].split("/")[-1]+".jpg"
        split = dial["split"]
        dial["share_id"] = 0
        iid2split[filename] = split

        dialogue = dial["dialogue"]
        user_one = []
        user_zero = []
        temp_neg = []
        share = False
        idx = 0
        while idx < len(dialogue):
            while idx < len(dialogue) and dialogue[idx]["user_id"] == 1:
                user_one.append(dialogue[idx]["message"])
                idx += 1
            while idx < len(dialogue) and dialogue[idx]["user_id"] == 0:
                if dialogue[idx]["share_photo"] == True:
                    share = True
                if dialogue[idx]["message"]!='':
                    user_zero.append(dialogue[idx]["message"])
                idx += 1
            if share:
                # last_turn = temp_neg.pop()
                # iid2pos[filename].append(last_turn+" ".join(user_one+user_zero))
                last_turn = ""
                this_turn = " ".join(user_one+user_zero)
                if len(this_turn) < 30 and temp_neg != []:
                    last_turn = temp_neg.pop() + " "
                iid2pos[filename].append(last_turn + this_turn)
                share = False
                user_one = []
                user_zero = []
            else:
                temp_neg.append(" ".join(user_one+user_zero))
                share = False
                user_one = []
                user_zero = []
        iid2neg[filename].append(temp_neg)
        
    paths = list(glob(f"{root}/*/*.jpg"))
    random.shuffle(paths)

    caption_paths = [path for path in paths if path.split("/")[-1] in iid2pos]
    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2pos),
    )

    bs = [path2rest(path, iid2pos, iid2neg, iid2split) for path in tqdm(caption_paths)]
    del dialogs

    for split in ["train", "validation", "test"]:
        batches = [b for b in bs if b[-1] == split]
        print(f"{split} : ",len(batches))
        dataframe = pd.DataFrame(
            batches, columns=["image", "pos", "neg", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/photochat_intent_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del batches
        gc.collect()        

if __name__ == "__main__":
    root = "YOUR PATH"
    dataset_root = "/data/dataset"
    make_arrow(root, dataset_root)