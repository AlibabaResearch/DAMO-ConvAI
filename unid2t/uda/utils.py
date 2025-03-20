import logging
import json

from torch.utils.data import DataLoader


def write_text_file(lines, file_src):
    with open(file_src, 'w', encoding='utf-8') as f_out:
        for line in lines:
            f_out.write(line + '\n')


def read_text_file(file_src):
    with open(file_src, 'r', encoding='utf-8') as f_in:

        lines = f_in.readlines()
        lines = [line.strip() for line in lines]

        return lines


def write_json_file(dict_object, file_src):
    with open(file_src, 'w', encoding='utf-8') as f_out:
        json.dump(dict_object, f_out)


def write_to_json_file_by_line(data, file_src):
    with open(file_src, 'w') as f_out:
        for line in data:
            f_out.write(json.dumps(line) + '\n')


def init_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    return logging.getLogger()
