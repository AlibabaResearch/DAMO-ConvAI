import random, re, os
from data.prompt_dataset import *
from data.plot_dataset import *
from data.arxiv_dataset import *
from data.yelp_dataset import *
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from unidecode import unidecode
import functools
from rake_nltk import Rake
import urllib, sys
import urllib.request
import json, re
import numpy as np
from scipy.spatial.distance import cdist
from bert_serving.client import BertClient
from tqdm import trange
from random import shuffle


def compose(*functions):
    """ Executes a list of functions in order """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def prefix_truncate(window):
    """ truncates text to the prefix window size """

    def f(text):
        if len(text) > window:
            text = text[:window]
        return text

    return f


class Preprocessor_base():
    def __init__(self):
        self.fn = None

    def make_fn(self):
        raise NotImplementedError()

    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e


def encode_tuple(tokenizer, t):
    return tokenizer.encode(t[0]), tokenizer.encode(t[1]), tokenizer.encode(t[2])


def truncate_tuple(truncator, t):
    return truncator(t[0]), truncator(t[1]), truncator(t[2])


class Preprocessor(Preprocessor_base):
    def __init__(self, tokenizer, seq_len, data_type):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_type = data_type

    def make_fn(self):
        return compose(
            insert_keywords(self.tokenizer, self.data_type), 
            lambda input: encode_tuple(self.tokenizer, input) if isinstance(input, tuple) else [encode_tuple(self.tokenizer, inp) for inp in input],
            lambda input: truncate_tuple(prefix_truncate(self.seq_len), input) if isinstance(input, tuple) else [truncate_tuple(prefix_truncate(self.seq_len), inp) for inp in input]
        )



def extract_keywords(text, r):
    r.extract_keywords_from_text(text)
    # 114 2, +1 per 228, add one key per 2 sentences, which is 114 in length
    num = min(5, max(2, int(len(text) / 228.0 + 1.5)))
    key = [re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', k.strip('\'.,:?!;" ')) for k in r.get_ranked_phrases()[:num]]
    return key

# ? Insert special tokens?
def insert_keywords(tokenizer, data_type):
    def f(text_raw_dict):
        # 'prompt' in text_raw_dict --> wp dataset; 'title' in text_raw_dict --> wi dataset and other well preprocessed dataset
        summary = text_raw_dict['prompt'] if 'prompt' in text_raw_dict else text_raw_dict['title']
        story = text_raw_dict['story']

        if data_type == 't0':  # x, y, y
            if 'prompt' in text_raw_dict:
                pp = get_paragraph(story)
                story = '\n\n'.join(pp)
            else:
                pp = story.split('<newline><newline>')
                story = '\n\n'.join(pp)

            return summary + tokenizer.eos_token, story + tokenizer.eos_token, tokenizer.eos_token + story + tokenizer.eos_token
        elif data_type == 't1':  # x, x + y, x + y
            if 'prompt' in text_raw_dict:
                pp = get_paragraph(story)
                story = '\n\n'.join(pp)
            else:
                pp = story.split('<newline><newline>')
                story = '\n\n'.join(pp)

            summary_story = summary + tokenizer.eos_token + story + tokenizer.eos_token
            return summary + tokenizer.eos_token, summary_story, tokenizer.eos_token + summary_story
        elif data_type == 't2':  # x, x + o + y, x + o + y, append
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            story_appended = summary + ''.join(keys_str) + tokenizer.eos_token + '\n\n'.join(pp)
            return summary + tokenizer.eos_token, story_appended + tokenizer.eos_token, tokenizer.eos_token + story_appended + tokenizer.eos_token
        elif data_type == 't3':  # x, x + o + y, x + o + y, insert
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            keys_str[0] += tokenizer.eos_token
            story_inserted = summary + ''.join([k + pt for k, pt in zip(keys_str, pp)])
            return summary + tokenizer.eos_token, story_inserted + tokenizer.eos_token, tokenizer.eos_token + story_inserted + tokenizer.eos_token
        elif data_type == 't4':  # x + o, y, x + o + y
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            summary_story = tokenizer.eos_token + summary + ''.join(keys_str) + tokenizer.eos_token + story + tokenizer.eos_token
            return summary + ''.join(keys_str) + tokenizer.eos_token, story + tokenizer.eos_token, summary_story
        elif data_type == 't5':  # x + o, x + o + y, x + o + y, append
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            story_appended = summary + ''.join(keys_str) + tokenizer.eos_token + '\n\n'.join(pp)
            return summary + ''.join(keys_str) + tokenizer.eos_token, story_appended + tokenizer.eos_token, tokenizer.eos_token + story_appended + tokenizer.eos_token
        elif data_type == 't6':  # x + o, x + o + y, x + o + y, insert
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            keys_str[0] += tokenizer.eos_token
            story_inserted = summary + ''.join([k + pt for k, pt in zip(keys_str, pp)])
            return summary + ''.join(keys_str) + tokenizer.eos_token, story_inserted + tokenizer.eos_token, tokenizer.eos_token + story_inserted + tokenizer.eos_token
        elif data_type == 't7':  # x + o, x + o + y, x + o + y, append, extend
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]

            extended_res = []
            for i in range(len(pp)):
                k_i, p_i = keys_str[:i], pp[:i]
                out_i = summary + ''.join(k_i) + tokenizer.eos_token
                story_appended_i = summary + ''.join(k_i) + tokenizer.eos_token + '\n\n'.join(p_i) + tokenizer.eos_token
                story_i = tokenizer.eos_token + summary + ''.join(k_i) + tokenizer.eos_token + '\n\n'.join(p_i) + tokenizer.eos_token
                extended_res.append((out_i, story_appended_i, story_i))
            return extended_res
        elif data_type == 't8':  # x + o, x + o + y, x + o + y, insert, extend
            if 'title' in text_raw_dict:
                pp = story.split('<newline><newline>')
            else:
                pp = get_paragraph(story)

            story = '\n\n'.join(pp)

            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            keys_str[0] += tokenizer.eos_token

            extended_res = []
            for i in range(len(pp)):
                k_i, p_i = keys_str[:i], pp[:i]
                out_i = summary + ''.join(k_i).replace(tokenizer.eos_token, '') + tokenizer.eos_token
                story_inserted_i = summary + ''.join([k + pt for k, pt in zip(k_i, p_i)]) + tokenizer.eos_token
                story_i = tokenizer.eos_token + summary + ''.join([k + pt for k, pt in zip(k_i, p_i)]) + tokenizer.eos_token
                extended_res.append((out_i, story_inserted_i, story_i))
            return extended_res
        else:
            raise Exception('Data type not implemented.')

    return f


def collate_fn(samples):
    """ Creates a batch out of samples """
    # each sample=[source, target, ?]
    x_max_len = max(map(lambda s: len(s[0]), samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(ss[0]) + [0] * (x_max_len - len(ss[0])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([ss[0] + [50256] * (x_max_len - len(ss[0])) for ss in samples])

    max_len = max(map(lambda s: len(s[1]), samples))
    # Zero pad mask
    y_mask = torch.ByteTensor([[1] * len(ss[1]) + [0] * (max_len - len(ss[1])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    y = torch.LongTensor([ss[1] + [50256] * (max_len - len(ss[1])) for ss in samples])

    max_len = max(map(lambda s: len(s[2]), samples))
    # Zero pad mask
    input_mask = torch.ByteTensor([[1] * len(ip[2]) + [0] * (max_len - len(ip[2])) for ip in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    input = torch.LongTensor([ip[2] + [50256] * (max_len - len(ip[2])) for ip in samples])

    return x_mask, x, y_mask, y, input[:, :-1], input[:, 1:].contiguous(), input_mask[:, 1:]
         # x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask


def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, data_type='t0', num_workers=1, make_train=True, make_val=True, make_test=False):
    # data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, num_workers = args.data_dir, args.dataset, tokenizer, batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1], batch_schedule[-1][0], batch_schedule[-1][1], args.workers

    loaders = []
    if dataset_name == 'wp':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/train.wp_source'),
                os.path.join(data_dir, 'writingPrompts/train.wp_target'),
                train_preproc)
            if data_type == 't7' or data_type == 't8':
                d_train = [t for lt in d_train for t in lt]
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/valid.wp_source'),
                os.path.join(data_dir, 'writingPrompts/valid.wp_target'),
                val_preproc)
            if data_type == 't7' or data_type == 't8':
                d_val = [t for lt in d_val for t in lt]
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/test.wp_source'),
                os.path.join(data_dir, 'writingPrompts/test.wp_target'),
                test_preproc)
            if data_type == 't7' or data_type == 't8':
                d_test = [t for lt in d_test for t in lt]
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'wi':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        print('Loading wikiplot dataset...')
        data_plots = os.path.join(data_dir, 'wikiPlots/plots_paragraph')
        data_titles = os.path.join(data_dir, 'wikiPlots/titles')
        with open(data_plots, errors='ignore') as fp:
            plots = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()

        texts = [(t, p) for t, p in zip(titles, plots) if t.strip() != '' and p.strip() != '']
        print('Done.')
        train_text = texts[:int(len(texts) * 0.9)]
        val_text = texts[int(len(texts) * 0.9):int(len(texts) * 0.95)]
        test_text = texts[int(len(texts) * 0.95):]

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = PlotDataset(train_text, train_preproc)
            if data_type == 't7' or data_type == 't8':
                d_train = [t for lt in d_train for t in lt]
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = PlotDataset(val_text, val_preproc)
            if data_type == 't7' or data_type == 't8':
                d_val = [t for lt in d_val for t in lt]
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = PlotDataset(test_text, test_preproc)
            if data_type == 't7' or data_type == 't8':
                d_test = [t for lt in d_test for t in lt]
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'ax':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        print('Loading arxiv dataset...')
        data_abs = os.path.join(data_dir, 'arxiv/artificial intelligence_10047_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/artificial intelligence_10047_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        ai_data = [('ai', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        data_abs = os.path.join(data_dir, 'arxiv/computer vision_14582_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/computer vision_14582_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        cv_data = [('cv', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        data_abs = os.path.join(data_dir, 'arxiv/language generation_14514_15000_15_abs.txt')
        data_titles = os.path.join(data_dir, 'arxiv/language generation_14514_15000_15_title.txt')
        with open(data_abs, errors='ignore') as fp:
            abs = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()
        assert len(titles) == len(abs)
        lg_data = [('lg', t.strip(), p.strip()) for t, p in zip(titles, abs) if t.strip() != '' and p.strip() != '']

        texts = ai_data + cv_data + lg_data
        shuffle(texts)
        print('Done.')
        train_text = texts[:int(len(texts) * 0.9)]
        val_text = texts[int(len(texts) * 0.9):int(len(texts) * 0.95)]
        test_text = texts[int(len(texts) * 0.95):]

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = ArxivDataset(train_text, train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = ArxivDataset(val_text, val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = ArxivDataset(test_text, test_preproc)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'yp':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = YelpDataset(os.path.join(data_dir, 'yelp/yelp.train.txt'), train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = YelpDataset(os.path.join(data_dir, 'yelp/yelp.valid.txt'), val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = YelpDataset(os.path.join(data_dir, 'yelp/yelp.test.txt'), test_preproc)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    else:
        raise Exception('Invalid dataset')

    return loaders
