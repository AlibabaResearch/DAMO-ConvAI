from pace.datasets import MMDialIntentDataset
from .datamodule_base import BaseDataModule


class MMDialIntentDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MMDialIntentDataset

    @property
    def dataset_cls_no_false(self):
        return MMDialIntentDataset

    @property
    def dataset_name(self):
        return "mmdial_intent"
