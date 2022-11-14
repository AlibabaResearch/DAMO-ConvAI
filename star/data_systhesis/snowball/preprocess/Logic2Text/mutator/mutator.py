import json
import sqlite3
import os
import random
# from tqdm import tqdm
import json
import time
import signal
import multiprocessing
from multiprocessing import Manager
from warnings import simplefilter

from preprocess.Logic2Text.logictools.utils import *
from preprocess.Logic2Text.logictools.APIs import *

alpha = 0.5  # keyword replacement threshold
beta = 0.5  # columns replacement threshold
gamma = 0.6  # adding columns ...(idle because logic2text does not allow)
theta = 0.15  # number(reversed threshold)
omega = 0.2  # idle (maybe for text or content mutation)
mutate_iter_num = 50
simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # load nodes
    data_path = "/home/yzh2749/snowball_new/data/logic2text/raw/"
    all_data_path = data_path + "all_data_origin.json"
    data_nodes = load_node_from_json(all_data_path)

    # mutate all nodes
    mutation = {}
    for node, code in data_nodes:
        mutated_nodes = node.mutate(mutate_num_max=mutate_iter_num,
                                    alpha=0.5,
                                    beta=0.5,
                                    gamma=0.6,
                                    theta=0.15,
                                    omega=0.2)
        mutation[code] = {n.to_code() + ' = ' + code.split(' ')[-1]: n.to_nl() for n in mutated_nodes if
                          n.to_code() != node.to_code()}

    # write to json
    with open('mutated_all_data_small.json', 'w') as file:
        json.dump(mutation, file, indent=4)
