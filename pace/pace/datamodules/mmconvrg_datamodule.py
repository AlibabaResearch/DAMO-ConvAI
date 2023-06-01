from pace.datasets import MMConvRGDataset
from .datamodule_base import BaseDataModule


class MMConvRGDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MMConvRGDataset

    @property
    def dataset_name(self):
        return "mmconvrg"