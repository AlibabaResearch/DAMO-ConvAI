import json
from abc import ABC, abstractmethod
from typing import Tuple

from eval_agent.utils.datatypes import State


class BaseEnv(ABC):
    def __init__(
        self,
        thought_instruction_path: str,
        action_instruction_path: str,
        thought_icl_path: str,
        action_icl_path: str,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        with open(thought_instruction_path) as f:
            self.thought_instruction = f.read()
        self.thought_raw_icl = json.load(open(thought_icl_path))
        
        with open(action_instruction_path) as f:
            self.action_instruction = f.read()
        self.action_raw_icl = json.load(open(action_icl_path))

        self.icl_format = icl_format
        self.max_steps = max_steps


    @abstractmethod
    def step(self, thought_output: str, action_output: str) -> Tuple[str, State, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset_action(self, first_thought: str) -> Tuple[str, State]:
        pass