
import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images1 = torch.stack([sample['patch_image1'] for sample in samples], dim=0)
    #patch_images2 = torch.stack([sample['patch_image2'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images1,
            #"patch_images_2": patch_images2,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


class Simmc2Dataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

   
    def __getitem__(self, index):
        turn_id, scene_image1, response, info = self.dataset[index]

        if scene_image1:
            scene_image1 = Image.open(BytesIO(base64.urlsafe_b64decode(scene_image1)))
            patch_scene_image1 = self.patch_resize_transform(scene_image1)

        # No Image
        #patch_scene_image1 = torch.ones((3, self.patch_image_size, self.patch_image_size))
        
        patch_mask = torch.tensor([True])

        info_token_list = info.strip().split()
        src_info = ' '.join(info_token_list[-self.max_src_length+1:])

        response_token_list = response.strip().split()
        tgt_response = ' '.join(response_token_list[:self.max_tgt_length])

        src_item = self.encode_text(" {}".format(src_info))
        tgt_item = self.encode_text(" {}".format(tgt_response))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": turn_id,
            "source": src_item,
            "patch_image1": patch_scene_image1,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example
    
    '''
    def __getitem__(self, index):
        turn_id, scene_image1, response, info, bbox = self.dataset[index]

        if scene_image1:
            scene_image1 = Image.open(BytesIO(base64.urlsafe_b64decode(scene_image1)))
            patch_scene_image1 = self.patch_resize_transform(scene_image1)

        # encode input
        info_token_list = info.strip().split()
        src_info = ' '.join(info_token_list[-self.max_src_length+1:])

        prompt_item = self.encode_text(" {} region: ".format(src_info))
        bbox_item = self.encode_text(bbox, use_bpe=False)
        src_item = torch.cat([prompt_item, bbox_item])

        # encode response
        response_token_list = response.strip().split()
        tgt_response = ' '.join(response_token_list[:self.max_tgt_length])
        tgt_item = self.encode_text(" {}".format(tgt_response))

        # add special symbol
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item]) 

        example = {
            "id": turn_id,
            "source": src_item,
            "patch_image1": patch_scene_image1,
            "patch_mask": torch.tensor([True]),
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example
     '''

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
