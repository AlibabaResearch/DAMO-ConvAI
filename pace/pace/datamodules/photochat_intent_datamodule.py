from pace.datasets import PhotochatIntentDataset
from .datamodule_base import BaseDataModule


class PhotoChatIntentDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PhotochatIntentDataset

    @property
    def dataset_cls_no_false(self):
        return PhotochatIntentDataset

    @property
    def dataset_name(self):
        return "photochat_intent"
