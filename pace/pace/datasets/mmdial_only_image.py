import random
import torch
import io
import pyarrow as pa
import os

from PIL import Image
from pace.transforms import keys_to_transforms


class MMDialImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        image_only=True,
        image_list=None,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.image_only = image_only
        self.data_dir = data_dir

        self.table = image_list

    def __len__(self):
        return len(self.table)

    def get_image(self, index, image_key="image"):
        binary = self.table[index]
        image_bytes = io.BytesIO(binary)
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "raw_index": index,
        }

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in -> {e}")
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        return dict_batch
