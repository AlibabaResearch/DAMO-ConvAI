import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union, Any, Mapping, Iterable, Union, List, Callable

import openai
import tqdm
from openai import openai_object
import copy


def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm.tqdm(fp.readlines(), desc=f"Reading JSON lines from {filename}", unit="lines"):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex
            

def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))


def write_jsonlines(
    objs: Iterable[Mapping[str, Any]], filename: str, to_dict: Callable = lambda x: x
):
    """Writes a list of Python Mappings as jsonlines at the input file."""
    with open(filename, "w") as fp:
        for obj in tqdm.tqdm(objs, desc=f"Writing JSON lines at {filename}"):
            fp.write(json.dumps(to_dict(obj)))
            fp.write("\n")