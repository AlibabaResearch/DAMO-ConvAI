import json
import torch
from os.path import join, exists

from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
from prefetch_generator import BackgroundGenerator
import copy
from .metadata import FASHION_COLOR, FASHION_PATTERN, FASHION_SLEEVE_LENGTH, FURNITURE_BRAND, FURNITURE_COLOR, FURNITURE_CUSTOMER_RATING, FURNITURE_MATERIALS, FURNITURE_PRICE, fashion_meta_attrs, furniture_meta_attrs, available_sizes2st

NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57

FASHION_SPECIAL_TOKENS = [f"<@1{i:03}>" for i in range(NUM_FASHION_ITEMS)]
FURNITURE_SPECIAL_TOKENS = [f"<@2{i:03}>" for i in range(NUM_FURNITURE_ITEMS)]

MAX_NUM_OBJ_IN_SCENE = 141
OBJECT_INDICES = [f"<{i}>" for i in range(MAX_NUM_OBJ_IN_SCENE)]

START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
NO_COREF = "<NOCOREF>"

INTENT_TOKEN = '<INTENT>'

FASHION_TYPE = "<FAS_TYPE>"
FASHION_PRICE = "<FAS_PRICE>"
FASHION_CUSTOMERREVIEW = "<FAS_CUSTOMER_REVIEW>"
FASHION_BRAND = "<FAS_BRAND>"
FASHION_SIZE = "<FAS_SIZE>"
FASHION_PATTERN = "<FAS_PATTERN>"
FASHION_COLOR = "<FAS_COLOR>"
FASHION_SLEEVE_LENGTH = "<FAS_SLEEVE_LENGTH>"
FASHION_AVAILABLE_SIZE = "<FAS_AVAILABLE_SIZE>"

FURNITURE_TYPE = "<FUR_TYPE>"
FURNITURE_MATERIALS = "<FUR_MATERIALS>"
FURNITURE_PRICE = "<FUR_PRICE>"
FURNITURE_BRAND = "<FUR_BRAND>"
FURNITURE_CUSTOMER_RATING = "<FUR_CUSTOMER_RATING>"
FURNITURE_COLOR = "<FUR_COLOR>"

FASHION_TOKEN_MAP = {
    'type': FASHION_TYPE, 
    'price': FASHION_PRICE, 
    'customerReview': FASHION_CUSTOMERREVIEW, 
    'brand': FASHION_BRAND, 
    'size': FASHION_SIZE, 
    'pattern': FASHION_PATTERN, 
    'color': FASHION_COLOR, 
    'sleeveLength': FASHION_SLEEVE_LENGTH, 
    'availableSizes': FASHION_AVAILABLE_SIZE
}
FURNITURE_TOKEN_MAP = {
    'type': FURNITURE_TYPE, 
    'material': FURNITURE_MATERIALS, 
    'price': FURNITURE_PRICE, 
    'brand': FURNITURE_BRAND, 
    'customerRating': FURNITURE_CUSTOMER_RATING, 
    'color': FURNITURE_COLOR
}

FASHION_DST = "<INTENT><FAS_TYPE><FAS_PRICE><FAS_CUSTOMER_REVIEW><FAS_BRAND><FAS_SIZE><FAS_PATTERN><FAS_COLOR><FAS_SLEEVE_LENGTH><FAS_AVAILABLE_SIZE>"
FURNITURE_DST = "<INTENT><FUR_TYPE><FUR_MATERIALS><FUR_PRICE><FUR_BRAND><FUR_CUSTOMER_RATING><FUR_COLOR>"


def get_input_id(tokenizer, tokens):
    # 获取输入的token所对应的id的信息：长度可能不唯一
    return tokenizer(tokens).input_ids[1:-1]


def id_converter(tokenizer):
    ''' 获取Specical Token所对应的id信息'''
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class LineByLineTask2Dataset(Dataset):
    def __init__(self, input_file, tokenizer: PreTrainedTokenizer, all_objects_meta, eval=False):
        ''' 训练的输入数据集'''
        
        with open(input_file) as f_in:
            self.data = json.load(f_in)
        
        # Other tasks
        lines = []
        self.boxes = []  # 存储了原来的Object Bbox Position信息
        self.generation = []
        self.nocoref = []
        self.disambiguation_objects = []
        self.disambiguation_labels = []
        self.misc = [] 
        self.is_fashion = []

        corefs = []
        
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}

        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        for dialog in self.data:

            self.disambiguation_labels.append(dialog['disambiguation_label'])
            self.is_fashion.append(dialog['is_fashion'])
            self.boxes.append(dialog['bbox'])
            lines.append(dialog['input'])

            corefs.append([f'<{index}>' for index in dialog['reference_objects']])  # 解决任务2

            # if not eval:
            #     coref_object = []
            #     coref_object.extend(dialog['reference_objects'])
            #     coref_object.extend(dialog['disambiguation_objects'])
            #     corefs.append([f'<{index}>' for index in coref_object])
            # else:
            #     # corefs.append([f'<{index}>' for index in dialog['reference_objects']])
            #     corefs.append([f'<{index}>' for index in dialog['disambiguation_objects']])  # 解决任务1


        encode_text = tokenizer(lines, add_special_tokens=True)

        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask

        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]  # 获取NOCOREF_ID的id形式
        
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        
        for idx, tokenized_line in enumerate(self.examples):
            
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id == EOM_id]
            if EOM_indices:  # 判断其是否为空
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1
            
            self.nocoref.append((tl.index(nocoref_id), 1 if not corefs[idx] else 0))  # 判断是否存在Object指代
            line_labels = []

            if self.is_fashion[idx]:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index 因为scene token都是在Multimodal Token id的后面
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]

                        fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        temp['object_index'] = item_index
                        
                        for attr_name, attr_value in all_objects_meta[fashion_st].items():
                            if attr_name != 'available_sizes':
                                temp['misc_labels'][attr_name] = fashion_meta_attrs[attr_name].index(attr_value)
                            else:
                                temp['misc_labels'][attr_name] = [1 if x in attr_value else 0 for x in fashion_meta_attrs[attr_name]] # 因为avaliable size的gt可能不止一个所以使用的损失函数不太一样可能有两个
                                
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]
                        furniture_st = id2furniture_st[tl[i+1]]
                        
                        temp['is_fashion'] = False
                        temp['pos'] = pos  # 代表是第几个Object Info
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        temp['object_index'] = item_index
                        
                        for attr_name, attr_value in all_objects_meta[furniture_st].items():
                            temp['misc_labels'][attr_name] = furniture_meta_attrs[attr_name].index(attr_value)
                            
                        line_labels.append(temp)
                        
            self.misc.append(line_labels)
        


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return  torch.tensor(self.examples[i], dtype=torch.long), \
                torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.boxes[i], \
                self.misc[i], \
                self.nocoref[i], \
                torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \


def get_task2_dataset(args, tokenizer, all_objects_meta, eval=False):
    
    if not eval:
        dataset = LineByLineTask2Dataset(args.train_input_file, tokenizer, all_objects_meta, eval=eval)
    else:
        dataset = LineByLineTask2Dataset(args.eval_input_file, tokenizer, all_objects_meta, eval=eval)

    return dataset


class LineByLineTask1Dataset(Dataset):
    def __init__(self, input_file, tokenizer: PreTrainedTokenizer, all_objects_meta, eval=False, eval_disam_path=None):
        ''' 训练的输入数据集'''
        
        with open(input_file) as f_in:
            self.data = json.load(f_in)
        
        # Other tasks
        lines = []
        self.boxes = []  # 存储了原来的Object Bbox Position信息
        self.nocoref = []
        self.misc = [] 
        self.is_fashion = []
        self.dialog_ids = []
        self.turn_ids = []
        
        self.eval = eval
        
        corefs = []
        
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}

        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        for dialog in self.data:
            
            self.is_fashion.append(dialog['is_fashion'])
            self.boxes.append(dialog['bbox'])
            self.dialog_ids.append(dialog['dialogue_idx'])
            self.turn_ids.append(dialog['turn_idx'])
            lines.append(dialog['input'])

            corefs.append([f'<{index}>' for index in dialog['disambiguation_objects']])  # 解决任务1

            # if not eval:
            #     coref_object = []
            #     coref_object.extend(dialog['reference_objects'])
            #     coref_object.extend(dialog['disambiguation_objects'])
            #     corefs.append([f'<{index}>' for index in coref_object])
            # else:
            #     # corefs.append([f'<{index}>' for index in dialog['reference_objects']])
            #     corefs.append([f'<{index}>' for index in dialog['disambiguation_objects']])  # 解决任务1



        encode_text = tokenizer(lines, add_special_tokens=True)

        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask
        
        # 读取 任务 2 的预测结果
        if eval_disam_path is None:
            self.disam_label = [1 for i in range(len(self.examples))]
        else:
            print('EXIST disambiguation predict result: ', eval_disam_path)
            with open(join(eval_disam_path, 'simmc2.1_task2_disam_predicted.json')) as f_in:
                self.disam_label = json.load(f_in)


        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        
        for idx, tokenized_line in enumerate(self.examples):
            
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id == EOM_id]
            if EOM_indices:  # 判断其是否为空
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1
            
            line_labels = []

            if self.is_fashion[idx]:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index 因为scene token都是在Multimodal Token id的后面
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]

                        fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        temp['object_index'] = item_index
                        
                        for attr_name, attr_value in all_objects_meta[fashion_st].items():
                            if attr_name != 'available_sizes':
                                temp['misc_labels'][attr_name] = fashion_meta_attrs[attr_name].index(attr_value)
                            else:
                                temp['misc_labels'][attr_name] = [1 if x in attr_value else 0 for x in fashion_meta_attrs[attr_name]] # 因为avaliable size的gt可能不止一个所以使用的损失函数不太一样可能有两个
                                
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]
                        furniture_st = id2furniture_st[tl[i+1]]
                        
                        temp['is_fashion'] = False
                        temp['pos'] = pos  # 代表是第几个Object Info
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        temp['object_index'] = item_index
                        
                        for attr_name, attr_value in all_objects_meta[furniture_st].items():
                            temp['misc_labels'][attr_name] = furniture_meta_attrs[attr_name].index(attr_value)
                            
                        line_labels.append(temp)
                        
            self.misc.append(line_labels)
        


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        if self.eval:
            return  torch.tensor(self.examples[i], dtype=torch.long), \
                    torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.boxes[i], \
                    self.misc[i], \
                    self.disam_label[i], \
                    self.dialog_ids[i], \
                    self.turn_ids[i]
        else:
            return  torch.tensor(self.examples[i], dtype=torch.long), \
                    torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                    self.boxes[i], \
                    self.misc[i] 
                    

def get_task1_dataset(args, tokenizer, all_objects_meta, eval=False, pretrain=False):
    
    if pretrain:
        dataset = LineByLineTask1Dataset(args.pretrain_input_file, tokenizer, all_objects_meta, eval=eval)
        return dataset
    
    if not eval:
        dataset = LineByLineTask1Dataset(args.train_input_file, tokenizer, all_objects_meta, eval=eval)
    else:
        dataset = LineByLineTask1Dataset(args.eval_input_file, tokenizer, all_objects_meta, eval=eval, eval_disam_path=args.eval_disam_path)
    
    return dataset


class LineByLineDSTDataset(Dataset):
    def __init__(self, input_file, tokenizer: PreTrainedTokenizer, all_objects_meta, eval=False, fashion_slot_map=None, furniture_slot_map=None):
        ''' 训练的输入数据集'''
        
        with open(input_file) as f_in:
            self.data = json.load(f_in)
        
        # Other tasks
        lines = []
        self.boxes = []  # 存储了原来的Object Bbox Position信息
        self.generation = []
        self.nocoref = []
        self.disambiguation_objects = []
        self.disambiguation_labels = []
        self.misc = [] 
        self.intent = []
        self.slot_value = []
        self.is_fashion = []
        self.slot_data = []
        self.intent_data = []
        
        self.intent_template = [
            'REQUEST:GET', 
            'REQUEST:COMPARE', 
            'INFORM:REFINE', 
            'INFORM:GET', 
            'ASK:GET', 
            'INFORM:DISAMBIGUATE', 
            'REQUEST:ADD_TO_CART'
        ]
        self.fashion_slot = {
            'type': 0, 
            'price': 0, 
            'customerReview': 0, 
            'brand': 0, 
            'size': 0, 
            'pattern': 0, 
            'color': 0, 
            'sleeveLength': 0, 
            'availableSizes': 0
        }
        self.furniture_slot = {
            'type': 0, 
            'material': 0, 
            'price': 0, 
            'brand': 0, 
            'customerRating': 0, 
            'color': 0
        }
        self.available_size_data = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        
        corefs = []
        
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}

        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        for dialog in self.data:

            self.disambiguation_labels.append(dialog['disambiguation_label'])
            self.is_fashion.append(dialog['is_fashion'])
            self.boxes.append(dialog['bbox'])
            self.intent_data.append(dialog['intent'])
            self.slot_data.append(dialog['slot_values'])
            
            
            lines.append(dialog['input'])

            corefs.append([f'<{index}>' for index in dialog['reference_objects']])  # 解决任务2

            # if not eval:
            #     coref_object = []
            #     coref_object.extend(dialog['reference_objects'])
            #     coref_object.extend(dialog['disambiguation_objects'])
            #     corefs.append([f'<{index}>' for index in coref_object])
            # else:
            #     # corefs.append([f'<{index}>' for index in dialog['reference_objects']])
            #     corefs.append([f'<{index}>' for index in dialog['disambiguation_objects']])  # 解决任务1


        encode_text = tokenizer(lines, add_special_tokens=True)

        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask

        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]  # 获取NOCOREF_ID的id形式
        
        intent_id = get_input_id(tokenizer, INTENT_TOKEN)[0]
        
        
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        
        for idx, tokenized_line in enumerate(self.examples):
            
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id == EOM_id]
            if EOM_indices:  # 判断其是否为空
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1
            
            self.nocoref.append((tl.index(nocoref_id), 1 if not corefs[idx] else 0))  # 判断是否存在Object指代
            self.intent.append((tl.index(intent_id), self.intent_template.index(self.intent_data[idx])))
            
            line_labels = []

            if self.is_fashion[idx]:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index 因为scene token都是在Multimodal Token id的后面
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]

                        fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        
                        for attr_name, attr_value in all_objects_meta[fashion_st].items():
                            if attr_name != 'available_sizes':
                                temp['misc_labels'][attr_name] = fashion_meta_attrs[attr_name].index(attr_value)
                            else:
                                temp['misc_labels'][attr_name] = [1 if x in attr_value else 0 for x in fashion_meta_attrs[attr_name]] # 因为avaliable size的gt可能不止一个所以使用的损失函数不太一样可能有两个
                                
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i
                        item_index = id2index[token_id]
                        furniture_st = id2furniture_st[tl[i+1]]
                        
                        temp['is_fashion'] = False
                        temp['pos'] = pos  # 代表是第几个Object Info
                        temp['coref_label'] = 1 if item_index in corefs[idx] else 0
                        temp['misc_labels'] = dict()
                        
                        for attr_name, attr_value in all_objects_meta[furniture_st].items():
                            temp['misc_labels'][attr_name] = furniture_meta_attrs[attr_name].index(attr_value)
                            
                        line_labels.append(temp)
              
            self.misc.append(line_labels)
            
            line_slot_values = self.slot_data[idx]
            line_slot_map = dict()
            
            if self.is_fashion[idx]:    
                line_slot_map = copy.deepcopy(self.fashion_slot)
                for attr_name in line_slot_map.keys():
                    attr_index = tl.index(get_input_id(tokenizer, FASHION_TOKEN_MAP[attr_name])[0])
                    if attr_name != 'availableSizes':
                        line_slot_map[attr_name] = (attr_index, fashion_slot_map[attr_name].index(line_slot_values[attr_name]) + 1) if attr_name in line_slot_values.keys() else (attr_index, 0)
                    else:
                        line_slot_map[attr_name] = (attr_index, [1 if item_size in line_slot_values.get(attr_name, []) else 0 for item_size in self.available_size_data])
            else:
                line_slot_map = copy.deepcopy(self.furniture_slot)
                for attr_name in line_slot_map.keys():
                    attr_index = tl.index(get_input_id(tokenizer, FURNITURE_TOKEN_MAP[attr_name])[0])
                    line_slot_map[attr_name] = (attr_index, furniture_slot_map[attr_name].index(line_slot_values[attr_name]) + 1) if attr_name in line_slot_values.keys() else (attr_index, 0)

            self.slot_value.append(line_slot_map)
            

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return  torch.tensor(self.examples[i], dtype=torch.long), \
                torch.tensor(self.examples_attention_mask[i], dtype=torch.long), \
                self.boxes[i], \
                self.misc[i], \
                self.nocoref[i], \
                torch.tensor(self.disambiguation_labels[i], dtype=torch.long), \
                self.intent[i], \
                self.slot_value[i]


def get_dst_dataset(args, tokenizer, all_objects_meta, eval=False, fashion_slot_map=None, furniture_slot_map=None):
    
    if not eval:
        dataset = LineByLineDSTDataset(args.train_input_file, tokenizer, all_objects_meta, eval=eval, fashion_slot_map=fashion_slot_map, furniture_slot_map=furniture_slot_map)
    else:
        dataset = LineByLineDSTDataset(args.eval_input_file, tokenizer, all_objects_meta, eval=eval, fashion_slot_map=fashion_slot_map, furniture_slot_map=furniture_slot_map)

    return dataset
