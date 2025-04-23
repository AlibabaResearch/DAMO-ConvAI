import enum
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class State:
    """This should contains everything needed to continue the conversation.

    For example, the history of the conversation, the current task (success/failure) at each step, etc.
    """

    def __init__(
        self,
        reward: float = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: str = None,
    ):
        """
        The history should be a format like:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        """
        self.history: List[Dict[str, Any]] = []
        self.reward: float = reward
        self.finished: bool = finished
        self.success: bool = success
        self.terminate_reason: str = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0

    @classmethod
    def load_json(cls, json_dict: Dict[str, Any]):
        state = cls()
        state.history = json_dict['conversations']
        info = json_dict['meta']
        state.reward = info["reward"]
        state.finished = info["finished"]
        state.success = info["success"]
        state.terminate_reason = info["terminate_reason"]
        state.error = info["error"]
        state.steps = info["steps"]
        return state

    @property
    def empty(self):
        return len(self.history) == 0

    def to_dict(self, format="fastchat") -> Dict[str, Any]:
        if format == 'openai':
            history = deepcopy(self.history)
        elif format == 'fastchat':
            history = []
            for idx, conv in enumerate(self.history):
                if idx % 2 == 0:
                    assert conv['role'] == 'user'
                    history.append({
                        "from": "human",
                        "value": conv['content'].strip(),
                    })
                else:
                    assert conv['role'] == 'assistant'
                    history.append({
                        "from": "gpt",
                        "value": conv['content'].strip(),
                    })
        meta_info = {
            "steps": self.steps,
            "reward": self.reward,
            "finished": self.finished,
            "success": self.success,
            "terminate_reason": self.terminate_reason,
            "error": self.error,
        }
        res_dict = {
            "meta": meta_info,
            "conversations": history
        }
        return res_dict
