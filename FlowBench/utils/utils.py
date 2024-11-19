import concurrent
import re
import threading
from string import Template

import numpy as np
import pandas as pd
import json
import math
import pickle
import os
import sys
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import openai
import random
import logging
import time
import traceback
import hashlib
import time, random, datetime
random.seed(666)
def string_to_md5(string):
    md5_val = hashlib.md5(string.encode('utf8')).hexdigest()
    return md5_val

def generate_18_digit_id():
    random_number = random.randint(100000000, 999999999)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    id_string = str(timestamp) + str(random_number)
    return id_string

def get_node_description(flow_description, node_id):
    pattern = r"{node_id}[\[\{{>]([^>\]\}}]+)[\]\}}]".format(node_id=re.escape(node_id))
    match = re.search(pattern, flow_description)
    if match:
        return match.group(1)
    else:
        pattern = r"{node_id}[\[\{{>]([^>\]\}}]+)[\]\}}]".format(node_id=re.escape(node_id+"("))
        match = re.search(pattern, flow_description) 
        return match.group(1) if match else "Unknown"

class LineProcessor:
    def __init__(
        self,
        fin,
        fout,
        num_workers,
        start=None,
        end=None,
        subset_ids=None,
        resume=True,
        shuffle=False,
        key_id="id",
        customize_filters=None,
        skip_err=True,
    ):
        self.fin = fin
        self.fout = fout
        self.num_workers = num_workers
        self.start = start
        self.end = end
        self.subset_ids = subset_ids
        self.resume = resume
        self.shuffle = shuffle
        self.customized_filters = customize_filters
        self.key_id = key_id
        self.skip_err = skip_err

        self.lines: list = None
        self.json_list: list = None

        self.lock = threading.Lock()
        self.bar = None

    def save_json(self, _d):
        with self.lock:
            with open(self.fout, mode='a', encoding="utf-8") as fp:
                if _d is not None and "mas" in _d:
                    dialog = _d["mas"]["dialog"]
                    apis = _d["apis"]
                    trace = _d["mas"]["trace"]
                    
                    sample_d = {}
                    sample_d["id"] = generate_18_digit_id()
                    sample_d["flow_id"] = trace["flow_id"]
                    sample_d["user_instruction"] = trace["user_instruction"]
                    sample_d["datetime"] = trace["datetime"]
                    sample_d["system_prompt"] = trace["Assistant.msg-3.prompt.system"]
                    sample_d["apis"] = apis
                    sample_d["dialog"] = dialog
                    if "thought" in _d["mas"]:
                        sample_d["thought"] = _d["mas"]["thought"]
                    line = json.dumps(sample_d, ensure_ascii=False) + "\n"
                    fp.write(line)
                else:
                    line = json.dumps(_d, ensure_ascii=False) + '\n'
                    fp.write(line)

    def preprocess(self):
        
        lines = open(self.fin).readlines()
        print(lines[0])
        json_list = []
        for line in lines:
            print(line)
            json_list.append(json.loads(line))
        print("# json raw:", len(json_list))

        for d in json_list:
            assert self.key_id in d

        if self.subset_ids is not None:
            subset_ids = set(self.subset_ids)
            json_list = [d for d in json_list if d[self.key_id] in subset_ids]
            print("# json in subset", len(json_list))

        if self.start is not None and self.end is not None:
            json_list = json_list[self.start : self.end]
            print("# json start-to-end", len(json_list))
        
        if os.path.exists(self.fout):
            if self.resume:
                done_ids = set()  
                flow_id = "id"                
                with open(self.fout, 'r') as f:
                    for line in f:
                        d = json.loads(line)
                        done_ids.add(d[flow_id]) 

                print(done_ids)
                json_list = [d for d in json_list if d[flow_id] not in done_ids]

                print("# json not-processed", len(json_list))
            else:
                pass

        if self.customized_filters is not None:
            for filter_func in self.customized_filters:
                json_list = [d for d in json_list if filter_func(d)]
                print("# json customized_filtered", len(json_list))

        # shuffle
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(json_list)

        self.json_list = json_list

    def run(self, process_func):
        self.preprocess()

        bar = tqdm(total=len(self.json_list))

        def __process_func(d):
            try:
                _d = process_func(d, self.save_json)
            except Exception as err:
                _id = d[self.key_id]
                logging.error(f"样本处理失败, id={_id}")
                traceback.print_exc()
                if not self.skip_err:
                    raise err

            bar.update()
            return _d

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(__process_func, self.json_list))

        print("ALL DONE", "=" * 100)
        return results


def convert_api_to_gpt(api, add_response_in_desc=True, has_response=False):
    api = json.loads(json.dumps(api, ensure_ascii=False))
    if "apiCode" not in api:
        api = api["apis"][0]
    name = api["apiCode"]
    desc = api["description"]
    params = unify_json(api["parameters"])

    params_schema = {"type": "object", "properties": params}

    if add_response_in_desc:
        resp_desc = build_response_desc(api)
        if len(desc) > 0 and desc[-1] not in ".?!)];。？！）】；":
            desc += "。"

        desc += resp_desc

    func = {"name": name, "description": desc, "parameters": convert_json_schema(json.loads(json.dumps(params_schema, ensure_ascii=False)))}

    if has_response:
        resp_def = unify_json(api["response"]["data"])
        resp_def = unify_response_def(resp_def)
        resp_def = convert_json_schema(json.loads(json.dumps(resp_def)))
        func["response"] = resp_def

    return func


def unify_json(json_or_str):
    if type(json_or_str) in (int, float, dict, list):
        return json_or_str
    elif type(json_or_str) == str:
        try:
            d = json.loads(json_or_str)
            return d
        except Exception:
            return json_or_str
    else:
        raise ValueError(f"invalid type of {json_or_str}")


def unify_response_def(d):
    tp = d["type"]
    if tp == "object":
        if "properties" in d:
            # 清理其他内容
            return {"properties": d["properties"], "type": "object"}
        return d
    else:
        # 转成object
        return {"properties": {"result": d}, "type": "object"}


def convert_json_schema(schema):
    """
    主要区别：
    1. list改成array
    2. required改成schema字段
    3. 支持enum（原本没有，不用管）
    """
    tp = schema["type"]
    # desc = schema['description']
    if tp in ("number", "integer", "string", "boolean"):
        return schema
    elif tp == "object":
        if "properties" in schema:
            properties = schema["properties"]
            required = []
            for prop, child_schema in properties.items():
                # 提取required，并删掉下层required字段
                if "required" in child_schema:
                    if child_schema["required"]:
                        required.append(prop)
                    del child_schema["required"]
                # 递归处理
                properties[prop] = convert_json_schema(child_schema)
            schema["required"] = required
    elif tp == "list" or tp == "array":
        schema["type"] = "array"
        if "items" in schema:
            items_schema = schema["items"]
            schema["items"] = convert_json_schema(items_schema)

    else:
        raise ValueError()
    return schema


def build_response_desc(api):
    result_desc = "该工具的返回内容有："
    response_data = api["response"]["data"]
    tp = response_data["type"]
    if tp in ("integer", "string", "number") or (tp == "list" and response_data["items"]["type"] in ("integer", "string", "number")):
        if "description" in response_data:
            result_desc += response_data["description"]
    elif tp == "list" and response_data["items"]["type"] == "object":
        if "description" in response_data:
            result_desc += response_data["description"]
        field_descs = []
        for i, (k, v) in enumerate(response_data["items"]["properties"].items()):
            if "description" in v:
                desc = v["description"]
                if desc[-1] != "。":
                    desc = desc + "。"
                field_descs.append(f"({i + 1}) " + desc.strip())
        if len(field_descs) > 0:
            result_desc += " 其中字段有：" + "".join(field_descs)
    elif tp == "object":
        for i, (k, v) in enumerate(response_data["properties"].items()):
            if "description" in v:
                desc = v["description"]
                if len(desc) > 0 and desc[-1] != "。":
                    desc = desc + "。"
                result_desc += f"({i + 1}) " + desc.strip()
    else:
        result_desc += response_data["description"]
    return result_desc


def dump_api(api):
    s = json.dumps(api, ensure_ascii=False, indent=2)
    s = re.sub(r',\n\s+"description', ', "description', s)
    s = re.sub(r'"parameters": {\n\s+"type": "object",\n\s+"properties', '"parameters": {"type": "object", "properties', s)
    return s
