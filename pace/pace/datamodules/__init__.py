from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .photochat_datamodule import PhotoChatDataModule
from .photochat_intent_datamodule import PhotoChatIntentDataModule
from .mmconvdst_datamodule import MMConvDSTDataModule
from .mmconvrg_datamodule import MMConvRGDataModule
from .simmc2rg_datamodule import SIMMC2GenDataModule
from .simmc2dst_datamodule import SIMMC2DSTDataModule
from .mmdial_caption_datamodule import MMDialCaptionDataModule
from .mmdial_intent_datamodule import MMDialIntentDataModule
from .imagechat_datamodule import ImageChatDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "photochat": PhotoChatDataModule,
    "photochat_intent": PhotoChatIntentDataModule,
    "mmconvdst": MMConvDSTDataModule,
    "mmconvrg": MMConvRGDataModule,
    "simmc2rg": SIMMC2GenDataModule,
    'simmc2dst': SIMMC2DSTDataModule ,
    "mmdial_caps": MMDialCaptionDataModule,
    "mmdial_intent": MMDialIntentDataModule,
    'imagechat':ImageChatDataModule,
}
