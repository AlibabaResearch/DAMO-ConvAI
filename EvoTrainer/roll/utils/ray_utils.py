import os

import ray


@ray.remote
def get_visible_gpus(device_control_env_var: str):
    return os.environ.get(device_control_env_var, "").split(",")


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
