from pace.datasets import PhotochatDataset
from .datamodule_base import BaseDataModule


class PhotoChatDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PhotochatDataset

    @property
    def dataset_cls_no_false(self):
        return PhotochatDataset

    @property
    def dataset_name(self):
        return "photochat"
