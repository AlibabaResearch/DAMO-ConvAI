import os
import sys
import torch
import datasets
from datasets.packaged_modules.text.text import Text
from config.decorator import replace
#from datasets import load_dataset
datasets.logging.set_verbosity(datasets.logging.ERROR)
from dataclasses import dataclass
import data.Processor as Processor
from torch.nn.utils.rnn import pad_sequence
import pyarrow as pa
from datasets import Features, Sequence, Value
from transformers.trainer_pt_utils import IterableDatasetShard
sys.path.append(".")
import json
class IterDataSet(torch.utils.data.IterableDataset):
    def __init__(self,files,mode):
        super().__init__()
        self.files = files
        self.mode = mode
        self.rowcnt = self.estimate_samples(self.files)
    def __len__(self):
        return self.rowcnt
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_worker = 1
            worker_id = 0
        else:
            num_worker = worker_info.num_workers
            worker_id = worker_info.id
        cnt = 0
        for f in self.files:
            for line in open(f,encoding='utf-8'):
                if cnt % num_worker == worker_id:
                    yield {self.mode: line.rstrip("\n")}
    def estimate_samples(self,filelist):
        cnt = 0
        for file in filelist:
            cnt += self.row_count(file)
        return cnt
    def row_count(self,filename):
        f = open(filename, 'rb')
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
        return lines
    
class InputPipe:
    """
    base class to store and process dataset 
    """
    def __init__(self, model, train_args, task_args, mode, auto_tokenizer):
        self.input_header = getattr(train_args, mode + "_header")
        self.model_header = getattr(train_args, mode + "_model_header")
        self.train_args = train_args
        self.task_args = task_args
        self.mode = mode
        self.model = model
        self.initialized = True
        self.workers = self.train_args.dataloader_num_workers if self.train_args.dataloader_num_workers else 1
        mode_identifier = "train" if mode == "train" else "eval"
        self.input_files = self.get_filenames(getattr(train_args,mode_identifier + "_dir"))

        print(self.input_files)

        if train_args.datareader_streaming:
            self.ds = IterDataSet(self.input_files,mode_identifier)
        else: #Default
            # import `DataBuilder` (with `BuilderConfig`) from 'text' module to load dataset

            self.ds = datasets.load_dataset('text',data_files=self.input_files,encoding='utf-8',features=Features({mode_identifier:Value("string")}))["train"]
        self.feature_extractor = FeatureExtractor(self.input_header, self.model_header, self.model, self.train_args,
                                                  self.task_args, auto_tokenizer)
    def get_dataset(self):
        return self.ds
    
    def get_filenames(self,name):
        if os.path.isdir(name):
            return [name + "/" + i for i in os.listdir(name)]
        else:
            return [name]

class FeatureExtractor():
    """
    Feature extractor to process features from dataloader in train/eval mode 

    Args : 
        input_header (:obj:`str`): 
            of form ``train/eval_header``
        model_header (:obj:`str`): 
            of form ``tokenizer:header1:header2:data_type``
        model (:obj:`PretrainedModel`)
            loaded model with model config
        train_args (:class:`~transformers.TrainingArguments`, `optional`):
            general train arguments 
        tast_args (:class:`~transformers.TrainingArguments`, `optional`):
            specific task arguments 

    """
    def __init__(self, input_header, model_header, model, train_args, task_args, auto_tokenizer):
        self.input_header = dict((h,i) for i,h in enumerate(input_header.split(",")))
        self.model_header = model_header.split(",")
        self.model = model
        self.train_args = train_args
        self.task_args = task_args
        self.out_types,self.out_shapes,self.out_paddings,self.padding_values= {},{},{},{}
        self.processors = dict()
        self.tokenizer = auto_tokenizer
        self.init_proc()
        self.set_property()
    def __call__(self,x):
        return self.preprocess(x)
    
    def init_proc(self):
        """ initialize line processor, e.g, tokenizer processor """
        for mh in self.model_header:
            cols = mh.split(":")
            out_name = cols[-1]
            if out_name in self.processors:
                raise ValueError(f"processor for out_name={out_name} is already registered with Processor {self.processors[out_name]}")
            if len(cols) == 1:
                cls = getattr(Processor,'basic')
                self.processors[out_name] = cls(self.input_header[out_name],out_name)
            elif len(cols) == 2:
                cls = getattr(Processor,cols[0])
                self.processors[out_name] = cls(self.input_header[out_name],out_name)
            else:
                # out_name : input / doc 
                cls = getattr(Processor, cols[0])
                feat_idx = [self.input_header[x] for x in cols[1:-1]]
                self.processors[out_name] = cls(idx=feat_idx, out_name=out_name, model=self.model, cfg=self.train_args,
                                                task_cfg=self.task_args, tokenizer=self.tokenizer)
    def preprocess(self,line):
        """ Process input columns in batch form using e.g., tokenizer """
        if isinstance(line, dict):
            line = line['text']
        elif not isinstance(line, str):
            line = line.decode('utf-8')
        line = json.loads(line)
        column = [line['source'][0], line['target'][0], line['target'][0]]
        if len(column) != 3:
            print(line)

        res = dict()
        for n in self.processors:
            tmp = self.processors[n].process(column)
            if tmp == None:
                return None
            res.update(tmp)


        return res
    
    def set_property(self):
        """ 
        set padding values as map(`dict`) for e.g, "input_ids","attention_mask","labels" 
        """
        for n in self.processors:
            p = self.processors[n].property()
            self.padding_values.update(p["values"])

@dataclass
class CustomizeCollator:
    """
    Collator of train/eval dataloader, process and tensorize features if needed
    """
    train_dataset:InputPipe
    eval_dataset:InputPipe
    @property
    def feature_extractor(self):
        fn = {}
        fn["train"] = self.train_dataset.feature_extractor if self.train_dataset else None
        fn["eval"] = self.eval_dataset.feature_extractor if self.eval_dataset else None
        return fn

    def __call__(self, features):
        raw_features = []
        res = {}
        mode = None
        # get processed lines using tokenizer (feature_extractor)
        # features struct : {'train/eval': raw line}
        for sample in features:
            for key,line in sample.items():
                if mode == None:
                    mode = key
                raw_features.append(self.feature_extractor[key](line))

        # convert batch data (run in training) with padding data 
        for fn in raw_features[0]:
            # todo
            if fn in self.feature_extractor[mode].padding_values:
                if (isinstance(raw_features[0][fn], list) and isinstance(raw_features[0][fn][0], list)) or (hasattr(raw_features[0][fn], "shape") and len(raw_features[0][fn].shape)>=2):
                    fv = [torch.tensor(f[fn][i]) for f in raw_features for i in range(len(f[fn]))]
                else:
                    fv = [torch.tensor(f[fn]) for f in raw_features]
                if self.feature_extractor[mode].padding_values[fn] == None:
                    res[fn] = torch.stack(fv)
                else:
                    res[fn] = pad_sequence(fv,batch_first=True,padding_value=self.feature_extractor[mode].padding_values[fn])
            else:
                res[fn] = [f[fn] for f in raw_features]
        return res

def input_builder(model, train_args, task_args=None, tokenizer=None):
    train_pipe = InputPipe(model, train_args, task_args, 'train', tokenizer) if train_args.do_train else None
    eval_pipe = InputPipe(model, train_args, task_args, 'eval', tokenizer) if train_args.do_eval else None
    predict_pipe = InputPipe(model, train_args, task_args, 'predict', tokenizer) if train_args.do_predict else None
    return train_pipe, eval_pipe, predict_pipe




@replace(Text)
class Text(Text):
    def _generate_tables(self, files):
        schema = pa.schema(self.config.features.type if self.config.features is not None else {"text": pa.string()})
        for file_idx, file in enumerate(files):
            batch_idx = 0
            with open(file, "r", encoding=self.config.encoding) as f:
                while True:
                    batch = f.read(self.config.chunksize)
                    if not batch:
                        break
                    batch += f.readline()  # finish current line
                    batch = batch.strip("\n").split("\n")
                    pa_table = pa.Table.from_arrays([pa.array(batch)], schema=schema)
                    # Uncomment for debugging (will print the Arrow table size and elements)
                    yield (file_idx, batch_idx), pa_table
                    batch_idx += 1

@replace(IterableDatasetShard)
class IterableDatasetShard(IterableDatasetShard):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_examples = len(self.dataset)
    def __len__(self):

        return (len(self.dataset) - 1 - self.process_index) // self.num_processes + 1

if __name__ == "__main__":
    from config.parse_args import *
    base_args,train_args,model_args,task_args = parse_args()
    test = InputPipe("bert-base-uncased",train_args,'eval')
    ds = test.get_dataset()


