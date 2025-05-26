import re
import json
import logging
from typing import Tuple

from eval_agent.envs import BaseEnv
from eval_agent.tasks import WebShopTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State
from envs.webshop.src.webshop.web_agent_site.envs import WebAgentTextEnv


logger = logging.getLogger("agent_frame")


class WebShopEnv(BaseEnv):
    def __init__(
        self,
        task: WebShopTask,
        env: WebAgentTextEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: WebShopTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env
        
        self.thought_state = State()
        self.action_state = State()
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, thought_output: str, action_output: str) -> Tuple[str, State, State]:
        self.thought_state.history.append({
            "role": "assistant",
            "content": thought_output
        })
        self.action_state.history[-1]['content'] += action_output
        try:
            action = self.parse_action(action_output)
        except:
            observation = f"Environment Observation: Invalid format. The input must contains 'Action: '"
            self.thought_state.history.append({
                "role": "user",
                "content": f"Agent1's{action_output}{observation}",
            })
            self.action_state.history.append({
                "role": "user",
                "content": observation,
            })
            self.action_state.steps += 1
            self.action_state.reward = 0
            if self.action_state.steps >= self.max_steps:
                self.action_state.finished = True
                self.action_state.success = False
                self.action_state.terminate_reason = "max_steps"
                self.action_state.reward = 0
            return observation, self.thought_state, self.action_state
        try:
            observation, reward, done, info = self.env.step(action=action)
            observation = f"Environment Observation:\n{observation}"
        except AssertionError:
            observation = 'Environment Observation: Invalid action!'
            done = False

        self.thought_state.history.append({
            "role": "user",
            "content": f"Agent1's{action_output}{observation}",
        })
        self.action_state.history.append({
            "role": "user",
            "content": observation,
        })

        self.action_state.steps += 1
        if self.action_state.steps >= self.max_steps:
            self.action_state.finished = True
            self.action_state.success = False
            self.action_state.terminate_reason = "max_steps"
            self.action_state.reward = 0

        if done:
            self.action_state.finished = True
            self.action_state.success = True
            self.action_state.terminate_reason = "success"
            self.action_state.reward = reward

        return observation, self.thought_state, self.action_state
    
    def reset(self) -> Tuple[str, State]:
        self.thought_state = State()
        self.env.reset(self.session_id)
        cur_task = self.env.observation
        thought_prompt, thought_messages = prompt_with_icl(self.thought_instruction, self.thought_raw_icl, cur_task, 1)
        
        if self.icl_format == 'first':
            self.thought_state.history.append({
                "role": "user",
                "content": thought_prompt,
            })
        elif self.icl_format == 'conversation':
            self.thought_state.history = thought_messages
        
        return thought_prompt, self.thought_state

    def reset_action(self, first_thought: str) -> Tuple[str, State]:
        self.action_state = State()
        self.env.reset(self.session_id)
        cur_task = self.env.observation

        action_prompt, action_messages = prompt_with_icl(self.action_instruction, self.action_raw_icl, cur_task, 1)

        if self.icl_format == 'first':
            self.action_state.history.append({
                "role": "user",
                "content": action_prompt,
            })
            self.action_state.history.append({
                "role": "assistant",
                "content": f"{first_thought}",
            })
        elif self.icl_format == 'conversation':
            action_messages.append({
                "role": "assistant",
                "content": f"{first_thought}",
            })
            self.action_state.history = action_messages
        
        return action_prompt, self.action_state