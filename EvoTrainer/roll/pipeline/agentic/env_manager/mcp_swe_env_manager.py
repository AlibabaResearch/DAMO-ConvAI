import copy
import json
import os
import pickle
from contextlib import nullcontext
import time
from datetime import datetime
from threading import Lock
from typing import Dict, Optional
import pathlib
import traceback

import numpy as np
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.env import REGISTERED_ENVS
from roll.pipeline.agentic.env.parse_action_utils import default_parser_llm_response
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.datasets.data_distributor import DataDistributorManager
from roll.utils.env_action_limiter import get_global_limiter
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from roll.utils.deepseekv32_tool_parser import DeepSeekV32ToolParser
from roll.pipeline.agentic.env_manager.encoding_dsv32 import encode_messages, parse_message_from_completion_text


class RolloutCache:
    def __init__(self, env_id, group_id, tag):
        self.tag = None
        self.env_id = env_id
        self.group_id = group_id
        self.tag = tag

        self.history = []
        self.frames = []

        self.truncated = False
        self.terminated = False
        self.step = 0

        self.reset_success = None
        self.meta_data = None  # 样本元信息
        self.final_reward = 0
        self.max_length_exceed = False
        self.max_turn_exceed = False
        self.tool_failed = False
        self.reward_info = {
            'tool_call_reward': 0,
            'has_cot_reward': 0,
            'trajectory_reward': 0,
            'judge_info': ""
        }


# 以下情况reward为0
# - 对话轨迹过长
# - 超出规定的对话轮次

# 只有以下一种情况reward为-999
# - 获取最终reward失败
class MCPSweEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: Dict,
                 tokenizer: PreTrainedTokenizer,
                 reward_tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 reward_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 data_distributor: DataDistributorManager = None,
                 env_port=None,
                 *args, **kwargs):
        """
        """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.reward_tokenizer: PreTrainedTokenizer = reward_tokenizer
        self.tool_call_style = self.env_config.get("tool_call_style", "qwen3")
        if self.tool_call_style == "minimax":
            self.tool_parser = MinimaxM2ToolParser(self.tokenizer)
        elif self.tool_call_style == "glm":
            self.tool_parser = Glm47MoeModelToolParser(self.tokenizer)
        elif self.tool_call_style == "kimi":
            self.tool_parser = KimiK2ToolParser(self.tokenizer)
        elif self.tool_call_style == "deepseek_v32":
            self.tool_parser = DeepSeekV32ToolParser(self.tokenizer)
        else:
            self.tool_parser = Qwen3CoderToolParser(self.tokenizer)
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler = generate_scheduler
        self.reward_scheduler = reward_scheduler
        self.data_distributor = data_distributor

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.current_step = -1
        self.running = False

        self.reward_proxy = None
        if self.reward_scheduler:
            self.logger.info(f"reward_proxy: {self.worker_config.llm_proxy} {self.reward_scheduler}")
            self.reward_proxy: BaseLLMProxy = create_llm_proxy(
                generate_scheduler=self.reward_scheduler,
                llm_proxy_config=self.worker_config.llm_proxy,
                tokenizer=self.reward_tokenizer,
                available_actions=None
            )
        self.logger.info(f"reward_proxy: {self.worker_config.llm_proxy}")

        self.use_thread_lock = self.env_config.get("use_thread_lock", False)  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        with self.thread_lock:
            self.env = REGISTERED_ENVS[self.env_config['env_class']](self.env_config['config'],
                                                                     **{'env_port':env_port, 
                                                                     'reward_proxy': self.reward_proxy, 'env_config':self.env_config, 'tool_parser':self.tool_parser})

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        self.enable_thinking = self.env_config["enable_thinking"]

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            available_actions=None
        )

    def clean_text(self, obj):
        if isinstance(obj, str):
            return obj.encode("utf-8", "surrogateescape").decode("utf-8", "ignore")
        elif isinstance(obj, dict):
            return {k: self.clean_text(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_text(v) for v in obj]
        return obj

    def _save_experiment_info(self):
        """从环境变量读取 Git 信息，保存 experiment_info.txt 到 mcp/rollouts 同级目录"""
        git_branch = os.environ.get('GIT_BRANCH', '')
        git_commit_hash = os.environ.get('GIT_COMMIT_HASH', '')
        exp_name = os.environ.get('EXP_NAME', '')
        exp_base_dir = str(pathlib.Path(self.env.config.train_mcp_save_dir).parent.parent)
        exp_info_path = os.path.join(exp_base_dir, 'experiment_info.txt')

        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"# 实验信息\n"
            f"实验名称: {exp_name}\n"
            f"启动时间: {start_time}\n"
            f"代码分支: {git_branch}\n"
            f"Commit Hash: {git_commit_hash}\n"
            f"\n"
            f"# 实验成功后，使用以下命令创建 experiment 分支（在本地代码仓库执行）\n"
            f"git checkout -b experiment/user/{exp_name} {git_commit_hash}\n"
            f"git push origin experiment/user/{exp_name}\n"
        )
        try:
            pathlib.Path(exp_info_path).parent.mkdir(parents=True, exist_ok=True)
            with open(exp_info_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Saved experiment info to {exp_info_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save experiment info: {e}")

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        # 首次运行时保存实验信息到实验根目录
        self._save_experiment_info()

        log_stats = {"reset_time": [], "generate_time": [], "step_time": [], "current_step": [], "formulate_rollouts_time": []}
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        start_timestamp = time.time()
        # 初始化环境
        self.rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        while self.running and self.rollout_cache is not None:
            # 生成模型决策
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(self.rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            # 执行环境step
            if self.rollout_cache.reset_success:
                with Timer(name="step", logger=None) as step_timer:
                    if stop_reason == GenerateStopReason.FINISH:
                        self.rollout_cache: RolloutCache = self.step(lm_output)
                log_stats["step_time"].append(step_timer.last)
            else:
                self.rollout_cache.terminated = True
                self.rollout_cache.truncated = True

            if self.running and (self.rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.info(
                    f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                if stop_reason == GenerateStopReason.MAX_LENGTH:
                    self.rollout_cache.max_length_exceed = True
                    self.rollout_cache.terminated = True
                    self.rollout_cache.truncated = True
                
                # 形成rollout
                with Timer(name="formulate_rollouts", logger=None) as formulate_rollouts_timer:
                    rollout: DataProto = self.formulate_rollouts(self.rollout_cache)
                log_stats["formulate_rollouts_time"].append(formulate_rollouts_timer.last)
                
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0],
                                                                     dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                mcp_path = os.path.join(self.env.config.train_mcp_save_dir, str(self.mode), str(self.episode_id),
                                        f'{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.rollout_cache.env_id}_{timestamp}.pkl')
                pathlib.Path(mcp_path).parent.mkdir(parents=True, exist_ok=True)

                # 清理环境资源
                try:
                    self.env.clear()
                except Exception:
                    pass

                # 保存mcp相关的调用
                end_timestamp = time.time()
                duration_time = end_timestamp - start_timestamp
                start_timestamp_readable = datetime.fromtimestamp(start_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                end_timestamp_readable = datetime.fromtimestamp(end_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                try:
                    if hasattr(self.env, 'mcp_logs'):
                        mcp_info = {
                            'start_step': start_step,
                            'tag': self.rollout_cache.tag,
                            'group_id': self.rollout_cache.group_id,
                            'env_id': self.rollout_cache.env_id,
                            'episode_id': self.episode_id,
                            'group_seed': self.group_seed,
                            'start_timestamp': start_timestamp_readable,
                            'end_timestamp': end_timestamp_readable,
                            'duration_time': duration_time,
                            'detailed_times': log_stats,
                            'mcp_logs': self.env.mcp_logs,
                        }
                        with open(mcp_path, 'wb') as fw:
                            pickle.dump(mcp_info, fw)
                        self.logger.info(f"Saved mcp info to {mcp_path}")
                except Exception as e:
                    self.logger.info(f"Failed to save mcps: {e}")

                log_stats = {"reset_time": [], "generate_time": [], "step_time": [], "current_step": [], "formulate_rollouts_time": []}
                self.rollout_cache = None
                start_timestamp = time.time()
                self.rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None))

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config['group_id']))
        if self.episode_id is None:
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id

        # 从数据分发器获取数据, 如果不存在数据分片则返回None
        meta_data = self.data_distributor.get_data_for_worker(self.env_config['tag'], self.env_config['group_id'],
                                                              self.episode_id)
        if self.max_env_step_concurrent > 0:
            acquire_id = self.env_step_limiter.acquire()
            initial_message, reset_success = self.env.reset(meta_data=meta_data, seed=seed, **{'mode': self.mode})
            self.env_step_limiter.release(acquire_id)
        else:
            initial_message, reset_success = self.env.reset(meta_data=meta_data, seed=seed, **{'mode': self.mode})

        self.rollout_cache.meta_data = meta_data
        if reset_success:
            self.rollout_cache.reset_success = 1.0
        else:
            self.rollout_cache.reset_success = 0.0
        self.rollout_cache.history.append(
            {
                "messages": initial_message,
                "parsed_messages": copy.copy(initial_message)
            }
        )

        return self.rollout_cache

    def step(self, llm_output: DataProto):
        # 适用于通用场景下的模型返回的原始结果,
        if self.tool_call_style == "deepseek_v32":
            response = self.tokenizer.batch_decode(
                llm_output.batch['responses'])[0]
        else:
            response = self.tokenizer.batch_decode(
                llm_output.batch['responses'],
                skip_special_tokens=True
            )[0]
        # print(f"=====LLM Response=====\n {response}")
        # 适用于 MCP-style tool call 解析模型返回的结果
        parsed_message = default_parser_llm_response(
            response=response,
            enable_thinking=self.enable_thinking,
            tool_parser=self.tool_parser,
            tools=self.env.config.TOOLS,
            format=self.tool_call_style,
        )
        print(f"=====parsed_message=====\n {parsed_message}")
        parsed_messages = [item for items in self.rollout_cache.history for item in items["parsed_messages"]]
        if self.max_env_step_concurrent > 0:
            acquire_id = self.env_step_limiter.acquire()
            observation, reward, penalty, terminated, truncated, turn_info, metrics = self.env.step(
                llm_response=response,
                meta_data=self.rollout_cache.meta_data,
                enable_thinking=self.enable_thinking,
                parsed_message=parsed_message,
                parsed_messages=parsed_messages,
                reward_tokenizer=self.reward_tokenizer,
                reward_proxy=self.reward_proxy,
                rollout_cache=self.rollout_cache,
                generating_args=self.pipeline_config.reward.generating_args,
            )
            self.env_step_limiter.release(acquire_id)
        else:
            observation, reward, penalty, terminated, truncated, turn_info, metrics = self.env.step(
                llm_response=response,
                meta_data=self.rollout_cache.meta_data,
                enable_thinking=self.enable_thinking,
                parsed_message=parsed_message,
                parsed_messages=parsed_messages,
                reward_tokenizer=self.reward_tokenizer,
                reward_proxy=self.reward_proxy,
                rollout_cache=self.rollout_cache,
                generating_args=self.pipeline_config.reward.generating_args,
            )

        # 根据环境执行结果处理当前轮次对话的结果
        self.rollout_cache.history[-1]["parsed_messages"].append(parsed_message)
        self.rollout_cache.history[-1]["reward"] = reward
        self.rollout_cache.history[-1]["penalty"] = penalty
        self.rollout_cache.history[-1]["info"] = turn_info
        self.rollout_cache.history[-1]["metrics"] = metrics

        self.rollout_cache.step += 1  # 一问一答算一轮
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env.config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
                self.rollout_cache.max_turn_exceed = True

        # 检测连续的工具调用错误: 如果检测到连续的工具调用错误两次，这条轨迹会被标记tool_failed=True
        continuous_penalty_count = 0
        pre_penalty = None
        for i in self.rollout_cache.history:
            cur_penalty = i.get('penalty', 0.0)
            if cur_penalty < 0:
                if pre_penalty is not None and pre_penalty < 0:
                    continuous_penalty_count += 1
                else:
                    continuous_penalty_count = 1  # 开始新的惩罚序列
            else:
                continuous_penalty_count = 0  # 重置计数

            if continuous_penalty_count >= 2:
                self.rollout_cache.tool_failed = True
                break
            pre_penalty = cur_penalty

        # # 将observation加入history, 开启下一轮对话
        if isinstance(observation, str):
            if turn_info.get('is_role_as_user', False):
                self.rollout_cache.history.append(
                    {
                        "messages": [
                            {"role": "user", "content": observation}
                        ],
                        "parsed_messages": [
                            {'role': 'user', 'content': observation}
                        ]
                    }
                )
            else:
                self.rollout_cache.history.append(
                    {
                        "messages": [
                            {"role": "user", "content": observation}
                        ],
                        "parsed_messages": [
                            {'role': 'tool', 'content': observation}
                        ]
                    }
                )
        elif isinstance(observation, list):
            messages, parsed_messages = [], []
            for obs in observation:
                messages.append({"role": "user", "content": obs})
                parsed_messages.append({'role': 'tool', 'content': obs})
            self.rollout_cache.history.append(
                {
                    "messages": messages,
                    "parsed_messages": parsed_messages
                }
            )
        else:
            self.logger.error(f"observation type error: {type(observation)}, observation: {observation}")
            self.rollout_cache.history.append(
                {
                    "messages": [
                        {"role": "user", "content": observation}
                    ],
                    "parsed_messages" : [
                        {'role': 'tool', 'content': observation}
                    ]
                }
            )

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        content = self.rollout_cache.history[-1]
        messages = content["messages"]
        if self.tool_call_style == "deepseek_v32":
            encode_config = dict(thinking_mode="thinking", drop_thinking=True, add_default_bos_token=True)
            print(f"len(messages): {len(messages)}")
            if len(messages) == 0:
                prompt_ids = []
            if messages[0]["role"] == "system":
                messages[0]["tools"] = self.env.config.TOOLS
                prompt = encode_messages(messages, **encode_config)
                # print(f"=====prompt1=====\n {prompt}")
                prompt_ids = self.tokenizer.encode(prompt)
            else:
                system_mock = [{"role": "system", "content": "",  "tools": self.env.config.TOOLS}]
                system_mock_prompt = encode_messages(system_mock, **encode_config)
                system_token_ids_mock = self.tokenizer.encode(system_mock_prompt)
                prompt = encode_messages(system_mock + messages, **encode_config)
                # print(f"=====prompt2=====\n {prompt}")
                token_ids = self.tokenizer.encode(prompt)
                # print(token_ids[len(system_token_ids_mock):])
                prompt_ids = token_ids[len(system_token_ids_mock):]
                prompt3 = self.tokenizer.decode(prompt_ids)
                # print(f"=====prompt3=====\n {prompt3}")
                

            # prompt = encode_messages(messages, **encode_config)
            # print(f"prompt v333:\n {prompt}")
            # prompt_ids = self.tokenizer.encode(prompt)
                print(f"=====prompt3=====\n {prompt3}")
        else:
            prompt_ids = custom_apply_chat_template(messages=messages, tools=self.env.config.TOOLS,
                                                tokenizer=self.tokenizer, add_generation_prompt=True,
                                                enable_thinking=self.enable_thinking, skip_mock_system_prompt=self.pipeline_config.skip_mock_system_prompt)
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])

        input_ids = history_token_ids + prompt_ids
        if len(input_ids) >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(input_ids)},"
                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length - input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]
        input_messages = [item for items in self.rollout_cache.history for item in items["parsed_messages"]]

        lm_output: DataProto = self.llm_proxy.generate(
            messages=input_messages,
            tools=self.env.config.TOOLS,
            lm_input=lm_input,
            generation_config=generation_config
        )

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()

        if "infer_logprobs" in lm_output.batch:
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["prompt_ids"] = prompt_ids
        content["response_ids"] = response_ids
        if self.tool_call_style == "deepseek_v32":
            content["messages"].append(
                {"role": "assistant", "content": self.tokenizer.decode(response_ids)})
        else:
            content["messages"].append(
                {"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # 如果最后一轮是环境的observation则舍弃
        raw_history = copy.copy(self.rollout_cache.history)
        if self.rollout_cache.history[-1]["messages"][-1]["role"] == "user":
            self.rollout_cache.history.pop(-1)
        raw_messages = [item for items in raw_history for item in items["messages"]]

        # 调用环境的obtain_outcome_reward方法获取轨迹奖励
        with self.env_step_limiter:
            final_reward, reward_info = self._get_outcome_reward(rollout_cache=self.rollout_cache)

        self.rollout_cache.final_reward = final_reward
        self.rollout_cache.reward_info = reward_info

        step_scores = []
        penalty = []
        for i in self.rollout_cache.history:
            step_scores.append(i.get('reward', 0.0))
            penalty.append(i.get('penalty', 0.0))

        episode_score = self.rollout_cache.final_reward
        episode_penalty = sum(penalty)

        token_ids = []
        prompt_masks = []
        response_masks = []
        token_penalty = []
        sequence_length = 0
        dialogue_turns = len(self.rollout_cache.history)
        infer_logprobs = []
        for items in self.rollout_cache.history:
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))
            token_penalty.extend(
                [0] * len(items["prompt_ids"]) + [items.get("penalty", 0.0)] * len(items["response_ids"]))
            sequence_length += len(items["prompt_ids"]) + len(items["response_ids"])
            if "infer_logprobs" in items:
                infer_logprobs.extend([0] * len(items["prompt_ids"]) + items["infer_logprobs"])

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        token_penalty_tensor = torch.tensor(token_penalty, dtype=torch.float32).unsqueeze(0)

        if 1 in response_masks:
            first_response_idx = response_masks.index(1)
            prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        else:
            prompt_masks = [1] * len(token_ids)

        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length,
                                  pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)
        token_penalty_tensor = pad_to_length(token_penalty_tensor, length=self.pipeline_config.sequence_length,
                                             pad_value=0)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
            "penalty": torch.Tensor([episode_penalty]),
            "token_penalty": token_penalty_tensor,
        })
        if len(infer_logprobs):
            infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
            infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
            lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

        final_reward = self.rollout_cache.final_reward if self.rollout_cache.final_reward != -999.0 else 0.0
        detailed_reward_info = self.rollout_cache.reward_info.get('detailed_reward_info', '')
        judge_info = self.rollout_cache.reward_info.get('judge_info', '')

        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "sequence_length": np.array([sequence_length], dtype=object),
            "dialogue_turns": np.array([dialogue_turns], dtype=object),
            "messages_list": np.array([raw_messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "step_scores": np.array([step_scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
            "final_reward": np.array([final_reward], dtype=object),
            "detailed_reward_info": np.array([detailed_reward_info], dtype=object),
            "judge_info": np.array([judge_info], dtype=object),
            "meta_data": np.array([self.rollout_cache.meta_data], dtype=object),
        })

        env_metric = {
            'success': float(self.rollout_cache.terminated and not self.rollout_cache.truncated),
            'num_actions': self.rollout_cache.step,
        }

        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get('metrics', {}).items():
                if k == 'success':
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(custom_metric[k])

        env_metric = {f"env/{self.rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        if self.rollout_cache.reset_success is not None:
            env_metric[f"env/{self.rollout_cache.tag}/reset_success"] = self.rollout_cache.reset_success
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self.mode == 'train':
            save_path = os.path.join(self.env.config.train_rollout_save_dir, str(self.mode), str(self.episode_id),
                                     f'{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.rollout_cache.env_id}_{timestamp}.pkl')
        else:
            save_path = os.path.join(self.env.config.train_rollout_save_dir, str(self.mode), str(self.current_step),
                                     f'{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.rollout_cache.env_id}_{timestamp}.pkl')
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            history_copy = []
            for history_item in self.rollout_cache.history:
                history_copy.append(copy.copy(history_item))
            dump_res = {
                'final_reward': final_reward,
                'detailed_reward_info': self.rollout_cache.reward_info.get('detailed_reward_info', ''),
                'judge_info': self.rollout_cache.reward_info.get('judge_info', ''),
                'meta_data': self.rollout_cache.meta_data,
                'history': history_copy,
            }
            with open(save_path, 'wb') as fw:
                pickle.dump(dump_res, fw)
            self.logger.info(f"Saved rollout info to {save_path}")
        except Exception as e:
            self.logger.info(f"Failed to save rollout: {e}")

        return lm_input

    def _get_outcome_reward(self, rollout_cache):
        final_reward = -999.0
        reward_info = {
            'meta_info': '',
            'detailed_reward_info': '',
            'judge_info': '',
        }

        try:
            # 调用obtain_outcome_reward方法
            final_reward, reward_info = self.env.obtain_outcome_reward(
                rollout_cache=rollout_cache,
                reward_tokenizer=self.reward_tokenizer,
                reward_proxy=self.reward_proxy,
                generating_args=self.pipeline_config.reward.generating_args,
                enable_thinking=self.enable_thinking,
            )
        except Exception as e:
            # 一般情况是调用llm judge出现异常，请求超时或者llm返回格式错误
            self.logger.info(f"Failed to obtain outcome reward: {e}")
            traceback.print_exc()

        return final_reward, reward_info
