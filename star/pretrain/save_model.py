import os
import torch
from _utils.utils_inbatch import *
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast,ElectraForPreTraining
from transformers import WEIGHTS_NAME, CONFIG_NAME

checkpoint_path = "checkpoints/pretrain/vanilla_11081_40.0%.pth"
Path('./pretrained/sss').mkdir(exist_ok=True, parents=True)
output_dir = "pretrained/sss"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

electra_config = ElectraConfig.from_pretrained(f'google/electra-large-discriminator')
discriminator = ElectraForPreTraining(electra_config)
load_part_model(checkpoint_path, discriminator, 'discriminator')
model_to_save = discriminator.module if hasattr(discriminator, 'module') else discriminator
tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-large-discriminator')

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)
