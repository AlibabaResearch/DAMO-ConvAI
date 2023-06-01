from pace.datasets import ImageChatDataset
from .datamodule_base import BaseDataModule


class ImageChatDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImageChatDataset

    @property
    def dataset_cls_no_false(self):
        return ImageChatDataset

    @property
    def dataset_name(self):
        return "imagechat"
