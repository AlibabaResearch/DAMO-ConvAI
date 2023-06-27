import os
import pickle
import torch.distributed as dist
from transformers import PretrainedConfig, WavLMConfig, RobertaConfig

CUSTOM_CONFIG_NAME = "config.json"
AUDIO_CONFIG_NAME = "audio_config.json"
TEXT_CONFIG_NAME = "text_config.json"


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def read_processed_pretrain(combined_path):
    if os.path.isdir(combined_path):
        datas = None
        for r, d, fs in os.walk(combined_path):
            if not d:
                for f in fs:
                    with open(os.path.join(r, f), "rb") as fp:
                        if datas is None:
                            datas = pickle.load(fp)
                        else:
                            datas += pickle.load(fp)
    else:
        with open(combined_path, "rb") as f:
            datas = pickle.load(f)
    return datas


class ATConfig(PretrainedConfig):
    audio_config_cls = WavLMConfig
    text_config_cls = RobertaConfig

    def __init__(self):
        super().__init__()
        self.text = self.audio_config_cls()
        self.audio = self.text_config_cls()

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs):
        config = cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME))
        if not return_unused_kwargs or len(kwargs) == 0:
            return config
        return config, kwargs

    @classmethod
    def from_configs(cls, audio, text):
        config = cls()
        config.audio = audio
        config.text = text
        return config

    @classmethod
    def from_classes(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_pretrained(audio), cls.text_config_cls.pretrained(text))

    @classmethod
    def from_json_files(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_json_file(audio), cls.text_config_cls.from_json_file(text))

    def set_pooling_mode(self, audio, text):
        self.text.pooling_mode = text
        self.audio.pooling_mode = audio

    def set_length(self, audio, text):
        self.text.max_length = text
        self.audio.max_length = audio


def get_ds_config(args, num_gpus):
    return {
        "train_batch_size": args.batch_size * num_gpus * args.grad_acc,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "zero_optimization": {
            "stage": args.ds_stage,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False
        },
        "fp16": {
            "enabled": True,
            "opt_level": f"O{args.apex_level}",
            "loss_scale_window": 200
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
