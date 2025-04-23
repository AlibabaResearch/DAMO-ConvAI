import os
import json
import yaml
import logging
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment.alfred_tw_env as envs

from eval_agent.tasks.base import Task


logger = logging.getLogger("agent_frame")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        env: envs.AlfredTWEnv,
        obs: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs

        self.env = env

    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1) -> Tuple[Iterable[Task], int]:
        os.environ["ALFWORLD_DATA"] = "eval_agent/data/alfworld"
        alfworld_data_path = os.environ.get("ALFWORLD_DATA")

        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)
        
        if split == 'train':
            split = "train"
            N_TASKS = 3321
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134

        env = getattr(alfworld.agents.environment.alfred_tw_env, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.alfred_tw_env.AlfredTWEnv)
        env = env.init_env(batch_size=1)

        if part_num > 1:
            assert part_idx != -1
            part_inst_num = [N_TASKS // part_num] * part_num
            part_inst_num[-1] += N_TASKS % part_num
            # jump to the start of the part
            env.skip(sum(part_inst_num[:part_idx]))
            N_TASKS = part_inst_num[part_idx]

        def generator():
            for idx in range(N_TASKS):
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                yield cls(
                    task_id=idx,
                    game_file=game_file,
                    env=env,
                    obs=obs,
                )

        return generator(), N_TASKS
