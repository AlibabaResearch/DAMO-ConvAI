from .base_gen_dataset import BaseGenDataset
from PIL import Image
import numpy as np
import base64

class MMConvRGDataset(BaseGenDataset):
    def __init__(self, 
            *args, 
            split="",
            **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            names = ["augment_mmconv_rg_train_0"] 
        elif split == "val":
            names = ["augment_mmconv_rg_val_0"]
        elif split == "test":
            names = ["rerank_augment_mmconv_rg_test_0"]

        # super().__init__(*args, **kwargs, names=names, source_column_name="prompt", target_column_name="text", text_column_name="caption")
        super().__init__(*args, **kwargs, names=names, text_column_name="source")

    def get_sep_token(self):
        return '[SEP]'
    
    #使用空白图片代替
    def get_raw_image(self, index, image_key="image"):
        img = Image.new('1', (300, 300), 1)
        return img.convert("RGB")