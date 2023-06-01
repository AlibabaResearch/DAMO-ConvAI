from .base_gen_dataset import BaseGenDataset
import io
from PIL import Image
import base64

class SIMMC2GenDataset(BaseGenDataset):
    def __init__(self, *args, split="" ,**kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            names = [f"simmc2.1_dev_{i}_rg" for i in range(50)] 
        elif split == "val":
            names = [f"simmc2.1_dev_{i}_rg" for i in range(6)]
        elif split == "test":
            names = [f"rerank_simmc2.1_devtest_{i}_rg_2turns" for i in range(12)]
        super().__init__(*args, **kwargs , names=names)
        self.turn_ids = self.table['turn_id'].to_pandas().tolist()

    def get_sep_token(self):
        return '[SEP]'
    
    def get_raw_image(self, index, image_key="image"):
        img_b64data = self.table[image_key][index].as_py()
        img_b64decode = base64.b64decode(img_b64data)
        img_bytes = io.BytesIO(img_b64decode)
        return Image.open(img_bytes).convert("RGB")

