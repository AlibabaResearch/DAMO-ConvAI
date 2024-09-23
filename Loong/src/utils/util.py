import os
import json
import logging
from colorlog import ColoredFormatter


def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def create_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def continue_gen(input_path, gen_data, tag):
    seen_id = dict()
    with open(input_path, 'r') as f:
        for item in f.readlines():
            js = json.loads(item.strip())
            if js[tag]:
                seen_id[js['id']] = js
    rewrite_data, continue_generate_data = [], []
    seen_rewrite = set()
    for item in gen_data:
        _id = item['id']
        if _id in seen_rewrite:
            continue
        if _id not in seen_id:
            continue_generate_data.append(item)
        else:
            rewrite_data.append(seen_id[_id])
        # dedup
        seen_rewrite.add(_id)
    with open(input_path, 'w') as f:
        for item in rewrite_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"continue_gen: input_path={input_path}, rewrite_data_num={len(rewrite_data)}, tag={tag}")
    return continue_generate_data



def setup_logger(name='Loong', level=logging.DEBUG):
    # create
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding repeatedly
    if not logger.hasHandlers():
        # log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # color
        formatter = ColoredFormatter(
            '%(log_color)s%(asctime)s (%(name)s - %(levelname)s)  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'green',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        # 将格式设置到处理器
        console_handler.setFormatter(formatter)

        # 将处理器添加到记录器
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()