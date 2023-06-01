from pace.datasets import VisualDialDataset
from .datamodule_base import BaseDataModule


class VisualDialDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VisualDialDataset

    @property
    def dataset_cls_no_false(self):
        return VisualDialDataset

    @property
    def dataset_name(self):
        return "visdial"