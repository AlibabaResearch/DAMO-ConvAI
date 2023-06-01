from pace.datasets import MMDialCaptionDataset
from pace.datasets import MMDialImageDataset
from .datamodule_base import BaseDataModule


class MMDialCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MMDialCaptionDataset

    @property
    def dataset_cls_no_false(self):
        return MMDialCaptionDataset
    
    @property
    def dataset_cls_only_image(self):
        return MMDialImageDataset

    @property
    def dataset_name(self):
        return "mmdial_caps"
