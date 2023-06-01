from .base_gen_dataset import BaseGenDataset
import io
from PIL import Image
import base64

class SIMMC2DSTDataset(BaseGenDataset):
    def __init__(self, *args, split="" ,**kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            names = [f"simmc2.1_train_{i}_dst" for i in range(40)] 
        elif split == "val":
            names = [f"simmc2.1_dev_{i}_dst" for i in range(40)]
        elif split == "test":
            names = [f"rerank_simmc2.1_devtest_{i}_dst_3turns" for i in range(5)]
        super().__init__(*args, **kwargs , names=names)

    def get_sep_token(self):
        return '[SEP]'
    
    #使用空白图片代替
    def get_raw_image(self, index, image_key="image"):
        img = Image.new('1', (300, 300), 1)
        return img.convert("RGB")

