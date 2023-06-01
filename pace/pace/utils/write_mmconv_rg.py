import json
import os
import pandas as pd
import pyarrow as pa
import random
import gc

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from copy import deepcopy
import re

class MMConvRGExtract():

    def get_token_text(self, token):
        return token.replace('<', '').replace('>', '').replace('|', '').strip()

    def call(self, text, begin_token, end_token=None, keep_tokens=False):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        end_idx = text.find(end_token)
        if begin_idx == -1:
            return ''
        elif end_idx == -1:
            return text[begin_idx + len(begin_token):].strip() if not keep_tokens else text[begin_idx:]
        return text[begin_idx + len(begin_token): end_idx].strip() if not keep_tokens else text[begin_idx: end_idx + len(end_token)]


class MMConvPreProcess():
    def __init__(self) -> None:
        self.remove_tokens={'<|imagesource|>': {'<|system|>', '<|user|>', '<|endofcontext|>', '<|endofresponse|>'}}
    
    def get_token_text(self, token):
        return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')

    def next_token(self, text):
        token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')
        result = token_matcher.search(text)
        return result if result is None else result[0]

    def extract(self, text, begin_token, end_token=None, no_token_in_between=True):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        if begin_idx == -1:
            return '', None
        begin_with_len = begin_idx + len(begin_token)
        end_idx = text[begin_with_len:].find(end_token)
        if end_idx == -1:
            return '', None
        end_idx += begin_with_len
        next_token_ = self.next_token(text[begin_with_len:])
        if not no_token_in_between or next_token_ == end_token:
            return text[begin_with_len: end_idx].strip(), begin_idx
        recurse_result = self.extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
        return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None

    def remove(self, text, begin_token, end_token=None, no_token_in_between=True, remove_begin_token=True, remove_end_token=True):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        if begin_idx == -1:
            return text
        begin_with_len = begin_idx + len(begin_token)
        end_idx = text[begin_with_len:].find(end_token)
        if end_idx == -1:
            return text
        end_idx += begin_with_len
        next_token_ = self.next_token(text[begin_with_len:])
        if not no_token_in_between or next_token_ == end_token:
            end_with_len = end_idx + len(end_token)
            return text[:(begin_idx if remove_begin_token else begin_with_len)].strip() + ' ' + text[(end_with_len if remove_end_token else end_idx):].strip()
        return text[:begin_with_len] + self.remove(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between, remove_begin_token=remove_begin_token, remove_end_token=remove_end_token)


def make_arrow(root, dataset_root, img_dataset):
    MMProcess = MMConvPreProcess()
    
    for split in ["train", "val", "test"]:
        raw_text = []
        bs = list()
        with open(f"{root}/{split}.simpletod") as f:
            data = [str(line.strip()) for line in f.readlines() if line.strip()]
        for i in tqdm(range(len(data))):
            raw_sample = data[i]
            for remove_token, end_tokens in MMProcess.remove_tokens.items():
                end_tokens = deepcopy(end_tokens)
            img_context = []
            while end_tokens:
                for end_token in list(end_tokens):
                    img_src, _ = MMProcess.extract(raw_sample, remove_token, end_token=end_token)
                    if not img_src:
                        end_tokens.discard(end_token)
                    else:
                        raw_sample = MMProcess.remove(raw_sample, remove_token, end_token=end_token, remove_end_token=False)
                        imgs = [img.strip() for img in img_src.split(",") if img_src!='']
                        img_context += imgs
            raw_text.append(raw_sample)
            context = [raw_sample]
            binary = []
            for im in img_context:
                with open(f"{img_prefix}/{im}", "rb") as fp:
                    img_io = fp.read()
                    binary.append(img_io)
            bs.append([binary, context, split])
        dataframe = pd.DataFrame(
            bs, columns=["image", "caption", "split"],
        )
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/mmconv_rg_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect() 
    print('SUCCESSFUL='*10)

if __name__ == "__main__":
    root="/data/MMConv/mt/resources"
    dataset_root="/data/dataset"
    img_prefix = "/data/MMConv/Image"
    make_arrow(root, dataset_root, img_prefix)

import json
import os
import pandas as pd
import pyarrow as pa
import random
import gc
import math
import collections
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from copy import deepcopy
import re
from transformers import BertTokenizer

class MMConvRGExtract():

    def get_token_text(self, token):
        return token.replace('<', '').replace('>', '').replace('|', '').strip()

    def call(self, text, begin_token, end_token=None, keep_tokens=False):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        end_idx = text.find(end_token)
        if begin_idx == -1:
            return ''
        elif end_idx == -1:
            return text[begin_idx + len(begin_token):].strip() if not keep_tokens else text[begin_idx:]
        return text[begin_idx + len(begin_token): end_idx].strip() if not keep_tokens else text[begin_idx: end_idx + len(end_token)]


class MMConvPreProcess():
    def __init__(self) -> None:
        self.remove_tokens={'<|imagesource|>': {'<|system|>', '<|user|>', '<|endofcontext|>', '<|endofresponse|>'}}
    
    def get_token_text(self, token):
        return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')

    def next_token(self, text):
        token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')
        result = token_matcher.search(text)
        return result if result is None else result[0]

    def extract(self, text, begin_token, end_token=None, no_token_in_between=True):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        if begin_idx == -1:
            return '', None
        begin_with_len = begin_idx + len(begin_token)
        end_idx = text[begin_with_len:].find(end_token)
        if end_idx == -1:
            return '', None
        end_idx += begin_with_len
        next_token_ = self.next_token(text[begin_with_len:])
        if not no_token_in_between or next_token_ == end_token:
            return text[begin_with_len: end_idx].strip(), begin_idx
        recurse_result = self.extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
        return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None

    def remove(self, text, begin_token, end_token=None, no_token_in_between=True, remove_begin_token=True, remove_end_token=True):
        end_token = end_token or f'<|endof{self.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        if begin_idx == -1:
            return text
        begin_with_len = begin_idx + len(begin_token)
        end_idx = text[begin_with_len:].find(end_token)
        if end_idx == -1:
            return text
        end_idx += begin_with_len
        next_token_ = self.next_token(text[begin_with_len:])
        if not no_token_in_between or next_token_ == end_token:
            end_with_len = end_idx + len(end_token)
            return text[:(begin_idx if remove_begin_token else begin_with_len)].strip() + ' ' + text[(end_with_len if remove_end_token else end_idx):].strip()
        return text[:begin_with_len] + self.remove(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between, remove_begin_token=remove_begin_token, remove_end_token=remove_end_token)


def make_arrow(root, dataset_root):
    MMProcess = MMConvPreProcess()
    img_prefix = "/data/downstream/Image"
    for split in ["train", "val", "test"]:
        raw_text = []
        bs = list()
        with open(f"{root}/{split}.simpletod") as f:
            data = [str(line.strip()) for line in f.readlines() if line.strip()]
        for i in tqdm(range(len(data))):
            raw_sample = data[i]
            for remove_token, end_tokens in MMProcess.remove_tokens.items():
                end_tokens = deepcopy(end_tokens)
            img_context = []
            while end_tokens:
                for end_token in list(end_tokens):
                    img_src, _ = MMProcess.extract(raw_sample, remove_token, end_token=end_token)
                    if not img_src:
                        end_tokens.discard(end_token)
                    else:
                        raw_sample = MMProcess.remove(raw_sample, remove_token, end_token=end_token, remove_end_token=False)
                        imgs = [img.strip() for img in img_src.split(",") if img_src!='']
                        img_context += imgs
            raw_text.append(raw_sample)
            context = [raw_sample]
            binary = []
            for im in img_context:
                with open(f"{img_prefix}/{im}", "rb") as fp:
                    img_io = fp.read()
                    binary.append(img_io)
            bs.append([binary, context, split])
        dataframe = pd.DataFrame(
            bs, columns=["image", "caption", "split"],
        )
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/mmconv_rg_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect() 
    print('SUCCESSFUL='*10)

def get_all_columns(dataset_root, names , columns=["image","source","target","split"]):
    tables = [
        pa.ipc.RecordBatchFileReader(
            pa.memory_map(f"{dataset_root}/{name}.arrow", "r")
        ).read_all()
        for name in names
        if os.path.isfile(f"{dataset_root}/{name}.arrow")
    ]
    table = pa.concat_tables(tables, promote=True)
    ret = {}
    for column in columns:
        ret[column] = table[column]
    return ret

def rewrite_arrow(dataset_root,output_dir,names,split):
    ret = get_all_columns(dataset_root ,names , columns=["image", "caption" , "split"])
    captions = ret['caption'].to_pandas().tolist()
    splits = ret['split'].to_pandas().tolist()
    images = ret['image']

    columns = ["image","source","target","split"]
    extracter = MMConvRGExtract()
    sources = list()
    targets = list()

    for caption in captions:
        source = extracter.call(caption[0], '<|context|>', keep_tokens=True)
        target = caption[0][len(source):]
        sources.append(source)
        targets.append(target)
    
    split_num = len(names)
    item_num = math.ceil(len(images)/split_num)

    tbar = tqdm(len(images))
    bs = list()
    for i in range(len(images)):
        bs.append([images[i].as_py() , sources[i], targets[i], splits[i]])
        tbar.update(1)
        if len(bs) % item_num == 0 or i+1 == len(images):
            j = math.ceil(i/item_num) - 1
            dataframe = pd.DataFrame(
                bs , columns=columns,
            )
            new_table = pa.Table.from_pandas(dataframe)
            bs = list()
            with pa.OSFile(
                f"{output_dir}/mmconv_rg_{split}_{j}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, new_table.schema) as writer:
                    writer.write_table(new_table)

#按照长度，升序排列
def rerank_samples_by_length(tokenizer , dataset_root , names):
    ret = get_all_columns(dataset_root ,names , columns=["image", "source", "target" , "split"])
    splits = ret['split'].to_pandas().tolist()
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    images = ret['image']
    source_lens = np.array([len(tokenizer.tokenize(sources[i])) for i in range(len(sources))])
    indexs = np.argsort(source_lens).tolist()
    columns = ["image","source","target","split"]
    new_sources = [sources[indexs[i]] for i in range(len(indexs))]
    new_targets = [targets[indexs[i]] for i in range(len(indexs))]
    new_images = [images[indexs[i]] for i in range(len(indexs))]
    new_splits = [splits[indexs[i]] for i in range(len(indexs))]

    split_num = len(names)
    item_num = math.ceil(len(images)/split_num)

    tbar = tqdm(len(images))
    bs = list()
    for i in range(len(images)):
        bs.append([new_images[i].as_py() , new_sources[i], new_targets[i], new_splits[i]])
        tbar.update(1)
        if len(bs) % item_num == 0 or i+1 == len(images):
            j = math.ceil(i/item_num) - 1
            dataframe = pd.DataFrame(
                bs , columns=columns,
            )
            new_table = pa.Table.from_pandas(dataframe)
            bs = list()
            with pa.OSFile(
                f"{dataset_root}/rerank_{names[j]}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, new_table.schema) as writer:
                    writer.write_table(new_table)
    print("rerank done")


def get_special_tokens(dataset_root , names):
    ret = get_all_columns(dataset_root ,names , columns=["image", "source", "target" , "split"])
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    texts = sources + targets
    special_tokens = list()
    patterns = [r'<\|[a-zA-Z]+\|>', r'[a-zA-Z]+_[a-zA-Z]+', r'\[[a-zA-Z]+\]' , r'[0-9.]+/[0-9]+' , r'[A-Za-z]+[/\&][A-Za-z]+',]
    for text in texts:
        for pattern in patterns:
            special_tokens.extend(re.findall(pattern , text))
    special_tokens = list(set(special_tokens))
    with open("../datamodules/vocabs/mmconv_special_tokens2.json","w") as f:
        json.dump(special_tokens,f)
    print("done")

def generate_vocab(extended_tokens_file , vocab_file, save_path):
    ex_id = 0
    with open(extended_tokens_file , "r" , encoding="utf-8") as f:
        extended_tokens = json.load(f)
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        if re.match(r'\[unused[0-9]+\]', token) != None:
            tokens[index] = extended_tokens[ex_id] + "\n"
            ex_id += 1
            if ex_id == len(extended_tokens):
                break
    
    assert ex_id == len(extended_tokens)

    with open(save_path, "w" , encoding='utf-8') as writer:
        writer.writelines(tokens)

def do_statistic(tokenizer , dataset_root , names):
    ret = get_all_columns(dataset_root ,names , columns=["image", "source", "target" , "split"])
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    source_lens = np.array([(len(tokenizer.tokenize(sources[i]))+2) for i in range(len(sources))])
    target_lens = np.array([(len(tokenizer.tokenize(targets[i]))+1) for i in range(len(targets))])
    texts = [sources[i] + " " + targets[i] for i in range(len(sources))]
    lens = np.array([(len(tokenizer.tokenize(texts[i]))+3) for i in range(len(texts))])
    print(f"mean len:{lens.mean()} , max len:{lens.max()}")
    print(f"source mean len:{source_lens.mean()} source max len:{source_lens.max()}")
    print(f"target mean len:{target_lens.mean()} target max len:{target_lens.max()}")

def augment_data(dataset_root , names , turn_nums=[0,-2,-4]):
    ret = get_all_columns(dataset_root ,names , columns=["image", "source", "target" , "split"])
    splits = ret['split'].to_pandas().tolist()
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    images = ret['image']
    columns = ["image","source","target","split"]

    def get_all_history_turns(text):
        return re.findall(r'(?:<\|[user|system]+\|>).+?(?=<)',text)

    new_items = []
    for i in range(len(images)):
        history_turns = get_all_history_turns(sources[i])
        for t in turn_nums:
            if len(history_turns) >= abs(t):
                new_items.append([images[i].as_py(), "<|context|>" + " ".join(history_turns[t:]) + "<|endofcontext|>", targets[i] , splits[i]])
    
    total_num = len(new_items)
    split_num = math.ceil(total_num/len(names))

    tbar = tqdm(len(names))
    for i in range(len(names)):
        dataframe = pd.DataFrame(new_items[i*split_num:(i+1)*split_num], columns=columns)
        new_table = pa.Table.from_pandas(dataframe)
        with pa.OSFile(
            f"{dataset_root}/augment_{names[i]}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, new_table.schema) as writer:
                writer.write_table(new_table)
        tbar.update(1)

def rewrite_rg_task_to_end2end(dataset_root , names):
    ret = get_all_columns(dataset_root ,names , columns=["image", "source", "target" , "split"])
    splits = ret['split'].to_pandas().tolist()
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    images = ret['image']
    columns = ["image","source","target","split"]
    def get_response_sec(text):
        m = re.search(r'<\|response\|>+.+?<\|endofresponse\|>',text)
        assert m != None
        start_pos ,end_pos = m.span()
        return text[start_pos:end_pos]
    
    new_items = []
    for i in range(len(images)):
        new_items.append([images[i].as_py() , sources[i], get_response_sec(targets[i]) , splits[i]])
    
    total_num = len(new_items)
    split_num = math.ceil(total_num/len(names))
    tbar = tqdm(len(names))
    for i in range(len(names)):
        dataframe = pd.DataFrame(new_items[i*split_num:(i+1)*split_num], columns=columns)
        new_table = pa.Table.from_pandas(dataframe)
        with pa.OSFile(
            f"{dataset_root}/end2end_{names[i]}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, new_table.schema) as writer:
                writer.write_table(new_table)
        tbar.update(1)
    

if __name__ == "__main__":
    root="/data/MMConv/mt/resources"
    dataset_root="/data/dataset"
    make_arrow(root, dataset_root)
    output_dir = "/data/datasets/"
    rewrite_arrow(dataset_root=dataset_root , output_dir=output_dir,split="train",names=["mmconv_rg_train"])
    rewrite_arrow(dataset_root=dataset_root, output_dir=output_dir,split="val",names=["mmconv_rg_val"])
    rewrite_arrow(dataset_root=dataset_root, output_dir=output_dir,split="test",names=["mmconv_rg_test"])
    # get_special_tokens(dataset_root , ["mmconv_rg_train_0" , "mmconv_rg_val_0"])
    # mmconv_vocab_file = "../datamodules/vocabs/mmconv_extended_vocab.txt"
    special_tokens_file ="../datamodules/vocabs/mmconv_special_tokens3.json"
    # generate_vocab(special_tokens_file, "../datamodules/vocabs/vocab.txt" , mmconv_vocab_file)
    with open(special_tokens_file , "r") as f:
        words = json.load(f)
    # tokenizer = BertTokenizer.from_pretrained(mmconv_vocab_file, do_lower_case=True, nerver_split=words)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer.add_special_tokens(words)
    print(len(tokenizer))
    do_statistic(tokenizer , dataset_root,["mmconv_rg_train_0" , "mmconv_rg_val_0"])
    augment_data(dataset_root , names=["mmconv_rg_train_0"])
    augment_data(dataset_root , names=["mmconv_rg_val_0"])
    # rewrite_rg_task_to_end2end(dataset_root , names=["mmconv_rg_train_0"])
    # rewrite_rg_task_to_end2end(dataset_root , names=["mmconv_rg_val_0"])
    # rewrite_rg_task_to_end2end(dataset_root , names=["augment_mmconv_rg_train_0"])
    # rewrite_rg_task_to_end2end(dataset_root , names=["augment_mmconv_rg_val_0"])
    # rewrite_rg_task_to_end2end(dataset_root , names=["rerank_mmconv_rg_test_0"])
    augment_data(dataset_root , names=["mmconv_rg_test_0"] , turn_nums=[-4])
    rerank_samples_by_length(tokenizer, dataset_root , ["augment_mmconv_rg_test_0"])
    