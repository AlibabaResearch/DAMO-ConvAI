from pace.datasets import SIMMC2DSTDataset
from .datamodule_base import BaseDataModule


class SIMMC2DSTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SIMMC2DSTDataset

    @property
    def dataset_cls_no_false(self):
        return SIMMC2DSTDataset

    @property
    def dataset_name(self):
        return "simmc2dst"
