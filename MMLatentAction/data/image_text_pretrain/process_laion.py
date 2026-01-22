import os
from datasets import load_dataset
from tqdm import tqdm
import json


size=100000

data = load_dataset("laion/conceptual-captions-12m-webdataset", split=f"train")
data = data.select(range(size))  # 只取前 100k


image_folder = "laion/conceptual-captions-12m-webdataset/images"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

converted_data = []

for da in tqdm(data):
    # print(da)
    try:
        json_data = {}
        # print(da)
        json_data["id"] = da["__key__"]

        json_data["image_path"] = f"{da['__key__']}.jpg"
        da["jpg"].save(os.path.join(image_folder, json_data["image_path"]))

        txt= da["txt"]
        json_data["txt"] = txt

        converted_data.append(json_data)
    except:
        continue

    if len(converted_data)==size:
        break

size=len(converted_data)


with open(f"laion/conceptual-captions-12m-webdataset/image_caption-{size}.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)