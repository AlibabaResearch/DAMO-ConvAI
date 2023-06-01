import pyarrow as pa
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import os
import gc
import io

'''
    ['image','history','style','answer','candidates','source','target','image_hash']
'''
def iid2content(dialog,imgs_root,split):
    image_hash =str(dialog['image_hash'])
    rounds = len(dialog['dialog'])
    turns = dialog['dialog']

    path = os.path.join(imgs_root,image_hash+".jpg")

    target_image_exist = False
    if os.path.exists(path):
        target_image_exist = True

    # image =''
    assert target_image_exist == True

    with open(path ,'rb') as vi:
        image = vi.read()
        image_bytes = io.BytesIO(image)
        image_bytes.seek(0)
        pimage = Image.open(image_bytes).convert("RGB")
    ret = list()
    history = list()
    #展开所有dialog
    #style是给定的！
    for turn_id,turn in enumerate(turns):
        style , utter = turn[0] , turn[1]
        source = history + [style + ":"]
        target = utter
        if 'candidates' in dialog:
            candidates = dialog['candidates'][turn_id]['100']
            for idx,cd in enumerate(candidates):
                if cd == utter:
                    gt_index = idx
                    break
            candidates[0],candidates[gt_index] = candidates[gt_index],candidates[0]
            
            ret.append([image,history.copy(),style,utter,candidates,source,target,image_hash])
        else:
            ret.append([image,history.copy(),style,utter,[utter],source,target,image_hash])
        history.append(style+":"+utter)
    return ret

def make_arrow(root,imgs_root,output_root):
    missed_dialogs = 0
    for split in ['train','valid','test']:
        with open(f"{root}/{split}.json",'r') as fb:
            dialogs = json.load(fb)
        
        sub_len = int(len(dialogs) // 8000)
        subs = list(range(sub_len + 1))
        for sub in tqdm(subs):
            reformed_dialogs =list()
            dialog_sub = dialogs[sub * 8000 : (sub + 1) * 8000]
            for dialog in tqdm(dialog_sub):
                try:
                    reformed_dialogs += iid2content(dialog,imgs_root,split)
                except Exception as e:
                    print(e)
                    with open("/data/error_idx.log",'a+') as f:
                        f.write(f"{split}\t{str(dialog)} \t Exception:{e} \n")
                    missed_dialogs += 1
            dataframe = pd.DataFrame(reformed_dialogs , columns=['image','history','style','answer',
                                                    'candidates','source','target','image_hash'])
            table = pa.Table.from_pandas(dataframe)
            os.makedirs(output_root, exist_ok=True)

            with pa.OSFile(
                f"{output_root}/imagechat_{split}_split_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del reformed_dialogs
            gc.collect()
    print(f"{missed_dialogs} totally is lost")

if __name__ =='__main__':
    input_root = "/data/datasets/imagechat"
    imgs_root = os.path.join(input_root,"imgs")
    output_root = "/data/datasets/"

    make_arrow(input_root, imgs_root , output_root)
