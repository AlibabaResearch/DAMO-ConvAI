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
import Levenshtein

def make_results(ignore_index, outputs, extras):
    span_pred = outputs['span'].detach().clone().argmax(dim=-1)
    gate_pred = outputs['gate'].detach().clone().argmax(dim=-1)
    action_pred = outputs['action'].detach().clone().argmax(dim=-1)
    slot_pred = outputs['slot']
    span_gt = extras['span']
    gate_gt = extras['gate']
    action_gt = extras['action']
    slot_gt = extras['slot_value']
    ids = extras['id'].cpu()
    input_ids_len = extras['input_ids_len']

    results = defaultdict(list)

    for i, span_gt_out in enumerate(span_gt):
        id2write = ids[i].item() if ids[i].nelement() == 1 else str(list(ids[i].numpy()))
        span_gt_out[span_gt_out == ignore_index] = 0
        span_gt_out = span_gt_out[:input_ids_len[i]].tolist()
        span_pred_out = span_pred[i][:len(span_gt_out)].tolist()
        gate_pred_out = gate_pred[i].tolist()
        gate_gt_out = gate_gt[i].tolist()
        action_gt_out = action_gt[i].item()
        action_pred_out = action_pred[i].item()
        slot_gt_out = slot_gt[i].item()
        if len(slot_pred[i].detach().clone()) == 0:
            slot_pred_out = -1
        else:
            slot_pred_out = slot_pred[i].detach().clone().argmax().item()

        predictions = {
            'ga': gate_pred_out,
            'os': span_pred_out,
            'ac': action_pred_out,
            'sl': slot_pred_out
        }
        gts = {
            'ga': gate_gt_out,
            'os': span_gt_out,
            'ac': action_gt_out,
            'sl': slot_gt_out
        }
        results[id2write].append({
            'predictions': predictions,
            'gts': gts
        })

    return results


remove_tokens={'<|imagesource|>': {'<|system|>', '<|user|>', '<|endofcontext|>', '<|endofresponse|>'}}

def gen_excerpts(text, nof_words=1):
    words = text.split()
    excerpts = set()
    for i in range(len(words) - nof_words + 1):
        excerpts.add(' '.join(words[i: i + nof_words]))
    return excerpts if excerpts else {text}
    
def levenshtein_ratio(len1, len2, dist):
    return (len1 + len2 - dist) / (len1 + len2)

def match(text, sub_text, thresh_abs=0, thresh_r=1, text_len_delta=[0, 0], return_thresh=1, sorted=True):
    base_split_len = len(sub_text.split())
    target_split_len = len(text.split())
    base_len = len(sub_text)
    good_matches = []
    lens = set()
    for delta in range(max(abs(text_len_delta[0]), abs(text_len_delta[1])) + 1):
        curr_lens = set()
        curr_lens.add(max(min(base_split_len - delta, target_split_len), 1, base_split_len + text_len_delta[0]))
        curr_lens.add(max(min(base_split_len + delta, target_split_len, base_split_len + text_len_delta[1]), 1))
        excerpts = set()
        for l in curr_lens.difference(lens):
            excerpts.update(gen_excerpts(text, nof_words=l))
        lens.update(curr_lens)
        matches = []
        for excerpt in excerpts:
            dist = Levenshtein.distance(sub_text, excerpt)
            if dist <= thresh_abs:
                matches.append([excerpt, dist])
        for m in matches:
            match_r = levenshtein_ratio(base_len, len(m[0]), m[1])
            if match_r >= thresh_r:
                good_matches.append(m + [match_r])
                if match_r >= return_thresh:
                    if sorted:
                        good_matches.sort(key=lambda m: m[-1], reverse=True)
                    return good_matches
    if sorted:
        good_matches.sort(key=lambda match: match[-1], reverse=True)
    return good_matches

def get_token_text(token):
    return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')

def next_token(text):
    token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')
    result = token_matcher.search(text)
    return result if result is None else result[0]

def extract(text, begin_token, end_token=None, no_token_in_between=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return '', None
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return '', None
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        return text[begin_with_len: end_idx].strip(), begin_idx
    recurse_result = extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
    return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None

def remove(text, begin_token, end_token=None, no_token_in_between=True, remove_begin_token=True, remove_end_token=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return text
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return text
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        end_with_len = end_idx + len(end_token)
        return text[:(begin_idx if remove_begin_token else begin_with_len)].strip() + ' ' + text[(end_with_len if remove_end_token else end_idx):].strip()
    return text[:begin_with_len] + remove(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between, remove_begin_token=remove_begin_token, remove_end_token=remove_end_token)


def make_arrow(root, dataset_root):
    img_prefix = "/data/downstream/Image"
    for split in ["train", "val", "test"]:
        bs = list()
        with open(f"{root}/{split}.dst") as f:
            data = [str(line.strip()) for line in f.readlines() if line.strip()]
        for i in tqdm(range(len(data))):
            raw_sample = data[i]
            for remove_token, end_tokens in remove_tokens.items():
                end_tokens = deepcopy(end_tokens)
            img_context = []
            while end_tokens:
                for end_token in list(end_tokens):
                    img_src, _ = extract(raw_sample, remove_token, end_token=end_token)
                    if not img_src:
                        end_tokens.discard(end_token)
                    else:
                        raw_sample = remove(raw_sample, remove_token, end_token=end_token, remove_end_token=False)
                        imgs = [img.strip() for img in img_src.split(",") if img_src!='']
                        img_context += imgs
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
            f"{dataset_root}/mmconv_dst_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect() 
    print('SUCCESSFUL='*10)

if __name__ == "__main__":
    root="/data/MMConv/dst/resources"
    dataset_root="/data/dataset"
    make_arrow(root, dataset_root)
