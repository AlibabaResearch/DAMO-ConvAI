import dataclasses
import functools
import os
from transformers import (
        HfArgumentParser,
        logging,
        AutoConfig
)
import config.ArgumentClass as ArgumentClass
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
def identifier(cls):
    return cls.__class__.__name__.split(".")[-1][:-6]

def print_arg(args):
    for k,v in args.__dict__.items():
        logger.info(k + ": " + str(v))

def print_args(argset):
    for setname,singlearg in argset.items():
        logger.info("<<<<<<<<"+setname+">>>>>>>>")
        print_arg(singlearg)


def parse_args():
    allargs = {}
    folder = os.path.dirname(os.path.abspath(__file__))
    #01. Base Configs
    parser = HfArgumentParser(ArgumentClass.base)#(getattr(ArgumentClass,'base')))
    allargs["base_args"],remain = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    #02. Data & Train Configs

    parser = HfArgumentParser(ArgumentClass.train)

    allargs["train_args"],remain_train = parser.parse_args_into_dataclasses(remain,args_filename= folder + "/SceneConfigs/"+allargs["base_args"].scene,return_remaining_strings=True)
    #03. Model Configs
    model_identifier = allargs["train_args"].model if not allargs["train_args"].model.startswith("local") else allargs["train_args"].previous_dir
    try:
        parser = HfArgumentParser((getattr(ArgumentClass,'model_' + allargs["train_args"].task)))
    except:
        logger.error("Task " + allargs["train_args"].task + " not registered!")
        exit(0)
    allargs["task_args"],remain_task = parser.parse_args_into_dataclasses(remain,args_filename=folder + "/SceneConfigs/"+allargs["base_args"].scene,return_remaining_strings=True)
    task_dict = dataclasses.asdict(allargs["task_args"])
    task_dict.update(allargs["task_args"].process())
    remain = [k for k in remain_task if k in remain_train and k != "http" and k[:2] == "--"]
    if remain:
        logger.error("unused command line args:" + str(remain))
        exit(1)
    allargs["model_args"],remain = AutoConfig.from_pretrained(model_identifier,return_unused_kwargs=True,_name_or_path=model_identifier,**task_dict)
    for k in remain:
        setattr(allargs["model_args"],k,remain[k])
    if remain:
        logger.warning("unused args:" + str(remain))
    if "DLTS_JOB_ID" in os.environ:
        log_dir = os.path.expanduser('~/tensorboard/{}/logs/'.format(os.environ['DLTS_JOB_ID']))
        allargs["train_args"].logging_dir = log_dir
        logger.info("Replace Tensorboard dir to " + log_dir)
    print_args(allargs)
    if not os.path.exists(allargs["train_args"].logging_dir) and allargs["train_args"].local_rank <= 0:
        os.makedirs(allargs["train_args"].logging_dir, exist_ok=True)
    if not os.path.exists(allargs["train_args"].output_dir) and allargs["train_args"].local_rank <= 0:
        os.makedirs(allargs["train_args"].output_dir, exist_ok=True)
    return allargs["base_args"],allargs["train_args"],allargs["model_args"],allargs["task_args"]
