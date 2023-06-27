import os
import torch
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

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                   os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME)), kwargs

    @classmethod
    def from_configs(cls, audio, text):
        config = cls()
        config.audio = audio
        config.text = text
        return config

    @classmethod
    def from_json_files(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_json_file(audio), cls.text_config_cls.from_json_file(text))

    def set_pooling_mode(self, audio, text):
        self.text.pooling_mode = text
        self.audio.pooling_mode = audio

    def set_length(self, audio, text):
        self.text.max_length = text
        self.audio.max_length = audio

        
class EarlyStopping:
    def __init__(self, mode='min', patience=10, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        self.best_epoch = 0
        
    def __call__(self, metric, epoch, model=None):
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func('Better performence.  Saving model')
        torch.save(model.state_dict(), self.path)
        self.best_score = score