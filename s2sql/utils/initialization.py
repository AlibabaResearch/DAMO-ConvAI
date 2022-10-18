#coding=utf8
""" Utility functions include:
    1. set output logging path
    2. set random seed for all libs
    3. select torch.device
"""

import sys, os, logging
import random, torch, dgl
import numpy as np

def set_logger(exp_path, testing=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

def set_random_seed(random_seed=999):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    dgl.random.seed(random_seed)

def set_torch_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two sentences are used to ensure reproducibility with cudnnbacken
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    return device
