import os
from datasets import load_dataset
from tqdm import tqdm
import json


size=100000

data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split=f"train")

image_folder = "lmms-lab/LLaVA-NeXT-Data/images"

converted_data = []


for da in tqdm(data):
    try:
        json_data = {}
        # print(da)
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image_path"] = f"{da['id']}.jpg"
            da["image"].save(os.path.join(image_folder, json_data["image_path"]))

        conversations= da["conversations"]
        json_data["conversations"] = conversations

        text=""
        for conversation in conversations:
            if conversation["from"]=="gpt":
                text+=conversation["value"]
                text+=" "

        json_data["text"] = text

        converted_data.append(json_data)
    except:
        continue


with open(f"lmms-lab/LLaVA-NeXT-Data/conversations-{size}.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)