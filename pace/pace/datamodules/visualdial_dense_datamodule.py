from pace.datasets import VisualDialDenseDataset
from .datamodule_base import BaseDataModule


class VisualDialDenseDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VisualDialDenseDataset

    @property
    def dataset_cls_no_false(self):
        return VisualDialDenseDataset

    @property
    def dataset_name(self):
        return "visdial_dense"
