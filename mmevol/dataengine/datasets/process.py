import json
import os
import os.path as osp
from tqdm import tqdm
import shutil

# Construct hash_id to create a unique index, because both id and image key values ​​have duplicate values
datasets_path = "/mnt/data/haonan/code/dataengine/datasets"

a = json.load(open(osp.join(datasets_path, "seed_data_1k_demo.json"), "r"))
for index, i in enumerate(a):
    i["hash_id"] = str(index) + "_" + i["image"].replace("/", "_")

json.dump(a, open("/mnt/data/haonan/code/dataengine/datasets/seed_data_1k_demo.json", "w"), indent=4)

# If the data format is already well organized, store it separately in meta data
if os.path.exists(osp.join(datasets_path, "meta_data")):
    shutil.rmtree(osp.join(datasets_path, "meta_data"))
    os.mkdir(osp.join(datasets_path, "meta_data"))

data = json.load(open(osp.join(datasets_path, "seed_data_1k_demo.json"), "r"))

for index, d in enumerate(tqdm(data)):
    json.dump(d, open(osp.join(datasets_path, "meta_data", "{}.json".format(d["hash_id"])), "w"), indent=4)