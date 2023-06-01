from pace.datasets import MMConvDSTDataset
from .datamodule_base import BaseDataModule


class MMConvDSTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MMConvDSTDataset

    @property
    def dataset_name(self):
        return "mmconvdst"