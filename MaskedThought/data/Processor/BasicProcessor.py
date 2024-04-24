from . import register_processor
from transformers import AutoTokenizer
import numpy
class BaseProcessor:
    def __init__(self, cfg, model, **kwargs):
        self.out_key = []
        self.padding_values = []
        tokenizer = kwargs.pop("tokenizer", None)
        self.fn = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model, cache_dir=cfg.cache_dir)
    def property(self):
        return {
                "values":dict([key,value] for key,value in zip(self.out_key,self.padding_values)) if self.padding_values else {}
                }
@register_processor("basic")
class SelectColumn(BaseProcessor):
    def __init__(self,idx,out_name,model=None,cfg = None,task_cfg=None):
        self.idx = idx
        self.out_key = [out_name]
        self.out_name = out_name
        self.padding_values = None

    def process(self,columns):
        try:
            return {self.out_name:columns[self.idx]}
        except IndexError as e:
            print(e)
            print('select', columns)
            return {self.out_name:""}

