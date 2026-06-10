import copy
import json
import time
from contextlib import nullcontext
from threading import Lock
from typing import Optional

import gem
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.pipeline.agentic.env.swe_env.utils import MultiprocessSafeLogger, write_data_json, pretty_print, Colors
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import GenerateStopReason
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import pad_to_length
from roll.utils.str_utils import contains_renderable_field

import logging
import os
from pprint import pprint
from dataclasses import asdict

loggers = {}


class TrajEnvManager(BaseEnvManager):
    def __init__(
        self,
        worker_config: EnvManagerConfig,
        pipeline_config: AgenticConfig,
        env_config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        generate_scheduler,
        output_queue: GroupQueueManager,
        thread_lock: Lock,
        mode="train",
        *args,
        **kwargs,
    ):
        """ """
        super().__init__()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config



        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RouterManager = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get(
            "use_thread_lock", False
        )  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # with self.thread_lock:
        if "seed" in self.env_config["config"]:
            self.env_config["config"]["seed"] = self.env_config["group_seed"]
        self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config["config"])

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        print("====== traj_env_manager_swe init ======")
        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template["agent_system_template"]
        self.agent_instance_template = cfg_template["agent_instance_template"]  # added for swe_env
        self.agent_obs_template = cfg_template["agent_obs_template"]  # added for swe_env
        self.agent_last_step_template = cfg_template["agent_last_step_template"]  # added for swe_env

        self.cur_seq_length = 0
        self.mode = self.pipeline_config.custom_envs[self.env_config["tag"]]["env_config"]["mode"]  # fix

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

        # tag_group_id_group_seed_env_id
        logger_name = f"{self.env_config['tag']}_env{self.env_config['env_id']}_group{self.env_config['group_id']}_seed{self.env_config['group_seed']}"
        self.logger = None  # 初始化 logger 为 None
        self.logger = MultiprocessSafeLogger(
            path=os.path.join(self.pipeline_config.base_dir, "log/traj_manager", f"traj_manager-{logger_name}.log")
        )

    def __del__(self):
        """析构函数，确保logger被正确关闭"""
        try:
            if hasattr(self, "logger") and self.logger:
                self.logger.close()
        except:
            pass  # 忽略析构函数中的异常

    def print_rollout_cache_structure(self, rollout_cache: Optional[RolloutCache] = None):
        """打印 rollout_cache 的数据结构"""
        if rollout_cache is None:
            rollout_cache = self.rollout_cache
        
        if rollout_cache is None:
            print("[ROLLOUT_CACHE] rollout_cache is None")
            return
        
        print("\n" + "="*80)
        print("[ROLLOUT_CACHE] 数据结构:")
        print("="*80)
        print(f"env_id: {rollout_cache.env_id}")
        print(f"group_id: {rollout_cache.group_id}")
        print(f"tag: {rollout_cache.tag}")
        print(f"step: {rollout_cache.step}")
        print(f"truncated: {rollout_cache.truncated}")
        print(f"terminated: {rollout_cache.terminated}")
        print(f"frames: {rollout_cache.frames}")
        print(f"history length: {len(rollout_cache.history)}")
        
        print("\n[ROLLOUT_CACHE] history 详细信息:")
        print("-"*80)
        for i, item in enumerate(rollout_cache.history):
            print(f"\n[History Item {i}]:")
            print(f"  Keys: {list(item.keys())}")
            # 打印每个键的基本信息
            for key, value in item.items():
                if key == "messages":
                    print(f"  {key}: list with {len(value)} messages")
                    for j, msg in enumerate(value):
                        role = msg.get("role", "unknown")
                        content_preview = str(msg.get("content", ""))[:100]
                        print(f"    [{j}] role={role}, content_preview={content_preview}...")
                elif key == "prompt_ids":
                    print(f"  {key}: list with {len(value)} token ids")
                elif key == "response_ids":
                    print(f"  {key}: list with {len(value)} token ids")
                elif key == "observation":
                    obs_preview = str(value)[:200] if value else "None"
                    print(f"  {key}: {obs_preview}...")
                elif key == "metrics":
                    print(f"  {key}: {value}")
                else:
                    value_str = str(value)
                    if len(value_str) > 100:
                        print(f"  {key}: {value_str[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        
        # 同时记录到日志
        if self.logger:
            self.logger.info(f"[ROLLOUT_CACHE] env_id={rollout_cache.env_id}, group_id={rollout_cache.group_id}, "
                           f"tag={rollout_cache.tag}, step={rollout_cache.step}, "
                           f"truncated={rollout_cache.truncated}, terminated={rollout_cache.terminated}, "
                           f"history_length={len(rollout_cache.history)}")


    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info["seed"] + self.env_config["group_seed"]
        print(f"[DEBUG][RUN_ROLLOUT_LOOP] group_seed: {self.group_seed}")
        
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step
        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        self.logger.info(
            f"[RUN_ROLLOUT_LOOP][ROLLOUT START][group_id: {self.env_config['group_id']}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id}, start_step: {start_step}]"
        )
        while self.running and rollout_cache is not None:
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)                        
            # 达到最大长度
            if stop_reason == GenerateStopReason.MAX_LENGTH: 
                rollout_cache.stop_reason = "reach_max_length"
            # 达到最大轮数
            elif rollout_cache.truncated: 
                rollout_cache.stop_reason = "reach_max_turn"
            # 主动提交
            elif rollout_cache.terminated: 
                rollout_cache.stop_reason = "self_submit"
            # 模型中断，会在什么情况出现？参数更新时会出现。
            elif stop_reason == GenerateStopReason.ABORT:
                rollout_cache.stop_reason = "abort"    
                print(f"[DEBUG][Attention]stop_reason == GenerateStopReason.ABORT ...")
            # 交互中
            elif stop_reason == GenerateStopReason.FINISH: # 交互中
                rollout_cache.stop_reason = "finish"
            else:
                rollout_cache.stop_reason = "unknown"
                print("[DEBUG][Attention]rollout_cache.stop_reason is unknown")
                print(f"[DEBUG]rollout_cache: \n{rollout_cache}")

            print(f"[RUN_ROLLOUT_LOOP]🟢 rollout_cache.stop_reason: {rollout_cache.stop_reason}, terminated: {rollout_cache.terminated}, truncated: {rollout_cache.truncated}, self.running: {self.running}")
            self.logger.info(f"[DEBUG][RUN_ROLLOUT_LOOP]rollout_cache.stop_reason: {rollout_cache.stop_reason}, terminated: {rollout_cache.terminated}, truncated: {rollout_cache.truncated}, self.running: {self.running}")
            if self.running and (
                rollout_cache.stop_reason in ["reach_max_length","reach_max_turn","self_submit"]
                or rollout_cache.terminated
                or rollout_cache.truncated
            ):
                self.logger.info(
                    f"[RUN_ROLLOUT_LOOP][STOP][group_id: {self.env_config['group_id']}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id}, start_step: {start_step}, gen_stats: {log_stats}, stop_reason: {rollout_cache.stop_reason}], terminated: {rollout_cache.terminated}, truncated: {rollout_cache.truncated}"
                )
                print(f"[RUN_ROLLOUT_LOOP][STOP]🟢 group_id: {self.env_config['group_id']}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id}, start_step: {start_step}, gen_stats: {log_stats}, stop_reason: {rollout_cache.stop_reason}], terminated: {rollout_cache.terminated}, truncated: {rollout_cache.truncated}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = (
                    f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                )
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array(
                    [traj_group_id] * rollout.batch.batch_size[0], dtype=object
                )
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                print(f"[traj_id]{traj_id} [traj_group_id]{traj_group_id} [episode_id]{self.episode_id} [group_seed]{self.group_seed} [env_id]{self.env_config['env_id']} [group_id]{self.env_config['group_id']} [tag]{self.env_config['tag']} [rollout.traj_id]{rollout.non_tensor_batch['traj_id']}")

                ray.get(
                    self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, rollout, self.env_config['env_id'])
                )
                # except Exception as e:
                #     print(f"[ERROR][FORMULATE_ROLLOUTS] error: {e}, rollout_cache: {rollout_cache}")
                #     self.logger.error(f"[ERROR][FORMULATE_ROLLOUTS] error: {e}, rollout_cache: {rollout_cache}")
                #     continue
                print(f"[RUN_ROLLOUT_LOOP]🟢 reset rollout_cache ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {start_step})")
                rollout_cache = self.reset()
                print(f"[RUN_ROLLOUT_LOOP]🟢 reset rollout_cache done ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {start_step})")
                start_step = self.current_step
        ray.get(self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, None, self.env_config['env_id']))

    def reset(self) -> RolloutCache:
        print(f"[MANAGER.reset]🟢 reset rollout_cache ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {self.current_step})")
        self.rollout_cache = RolloutCache(
            env_id=self.env_config["env_id"], group_id=self.env_config["group_id"], tag=self.env_config["tag"]
        )
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(
            self.env_config["group_id"],
            self.env_config["env_id"]
        ))
        print(f"[MANAGER.reset]🟢 get episode_id ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {self.current_step})")
        if self.episode_id is None:
            print(f"[MANAGER.reset]❌ get episode_id is None ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id})")
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id

        # reset env
        observation, info = self.env.reset(seed=seed, step=self.current_step)
        # print(f'\n\n[MANAGER.reset]\n******observation:******\n {observation}\n******info:******\n {info}')
        print(f'\n\n[MANAGER.reset]observation: {[observation]}, info: {info}')

        # log info
        task_idx = info.get("task_idx", None)
        traj_reset_time = info.get("traj_reset_time", None)
        logger_name = (f"{self.env_config['tag']}_task{task_idx}_reset{traj_reset_time}_env{self.env_config['env_id']}_group{self.env_config['group_id']}_seed{seed}")
        log_path = os.path.join(self.pipeline_config.base_dir, "log/traj_manager", f"{logger_name}.log")
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[manager.reset] tag: {self.env_config['tag']} task_idx: {task_idx} traj_reset_time: {traj_reset_time} (seed: {seed}, group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})")
        if self.logger is None: 
            self.logger = MultiprocessSafeLogger(path=log_path)
        else: 
            self.logger.update_log_path(path=log_path)
        self.logger.info(f"[manager.reset] tag: {self.env_config['tag']}, task_idx: {task_idx}, traj_reset_time: {traj_reset_time} (seed: {seed}, group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})")

        # append history
        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
                **info,
            }
        )
        if not observation:
            print(f"[MANAGER.reset]❌ observation is None ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {self.current_step})")
            # return "failed reset rollout_cache, observation is None."
            return None
        print(f"[MANAGER.reset]✅ success reset rollout_cache ... (group_id: {self.env_config['group_id']}, episode_id: {self.episode_id}, start_step: {self.current_step})")
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(llm_output.batch["responses"], skip_special_tokens=True)
        observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        # print(f'\n\n[MANAGER.step]\n******observation:******\n {observation}\nreward: {reward}\ntruncated: {truncated}\nterminated: {terminated}\n******info:******\n {info}')
        print(f'[MANAGER.step]{[observation[:100]]}...(trunced), reward: {reward}, truncated: {truncated}, terminated: {terminated}')

        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]["reward"] = reward
        self.rollout_cache.history[-1]["penalty"] = 0
        metrics = info.get("metrics", {})
        if not metrics.get("action_is_valid", True):
            self.rollout_cache.history[-1]["penalty"] = self.worker_config.format_penalty
        self.rollout_cache.history[-1]["llm_response"] = responses[0]
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            }
        )
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        """
        @input:
            llm_output: DataProto
        @output: (需要check一下)
            self.rollout_cache: 包含env_id=0, group_id=0, tag='SWEEnvTrain', history=[{},{}...], frames=[], truncated=False, terminated=True, step=7
            其中.history[-1] 包含reward, penalty, llm_response, suffix, metrics, info, observation, actions_left
        """
        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        content = self.rollout_cache.history[-1]
        render_dict = {"observation": content["observation"]}
        if contains_renderable_field(self.agent_obs_template, "turn_idx"):
            render_dict["turn_idx"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_obs_template, "suffix"):
            render_dict["suffix"] = content.get("suffix", "/testbed")
        if contains_renderable_field(self.agent_obs_template, "actions_left"):
            actions_left = min(content["actions_left"], (self.pipeline_config.sequence_length - self.cur_seq_length) // 512)
            render_dict["actions_left"] = actions_left
        if contains_renderable_field(self.agent_obs_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        
        # current messages
        messages = []
        system = self.agent_system_template
        user_first = self.agent_instance_template.format(problem_statement=content.get("observation", ""))
        if content.get("suffix"):
            system = system.replace("/testbed", content["suffix"])
            user_first = user_first.replace("/testbed", content["suffix"])
        if self.rollout_cache.step == 0:
            messages = [{"role": "system", "content": system}]
            messages.append({"role": "user", "content": user_first})
        else:
            messages.append({"role": "user", "content": self.agent_obs_template.format(**render_dict)})
        content["messages"] = messages
        prompt_ids = custom_apply_chat_template(
            messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True, skip_mock_system_prompt=self.pipeline_config.skip_mock_system_prompt
        )

        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        input_ids = history_token_ids + prompt_ids
        self.cur_seq_length = len(input_ids)

        # sequence length warining
        if len(input_ids) >= self.pipeline_config.sequence_length:
            print(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(input_ids)},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        # convert to tensor for lm_input
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        
        position_ids = attention_mask.cumsum(dim=-1) - 1
        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        # compute max_new_tokens
        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - input_ids.shape[1],
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        # print(f"\n****** history items start ***********\n")
        # for item in self.rollout_cache.history:
        #     print(f"item keys: {item.keys()}")
        # print(f'last item: \n{self.rollout_cache.history[-1]}')
        # print(f"\n****** history items end ***********")

        input_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        # roles = [item['role'] for item in input_messages]
        # print(f'\n\n[MAKE_DECISION]input_messages roles: {roles}')
        # for item in input_messages:
        #     print(f"\n{item}")
        # print(f'[MAKE_DECISION]******input_messages end ******')

        lm_output: DataProto = self.llm_proxy.generate(
            messages=input_messages, lm_input=lm_input, generation_config=generation_config
        )
        
        # 打印input_messages
        self.logger.info(f"[DEBUG][MAKE_DECISION](step:{self.rollout_cache.step})input_messages:")
        self.logger.info(f"input_messages length: {len(input_messages)}")
        self.logger.info(f"roles: [{[item['role'] for item in input_messages]}]\n")
        for item in input_messages:
            self.logger.info(f"\n{item}")
        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch["responses"][0]
        response_ids = response_ids.tolist()
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        content["prompt_ids"] = prompt_ids
        content["response_ids"] = response_ids

        if "infer_logprobs" in lm_output.batch.keys():
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["messages"].append(
            {"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)}
        )

        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """ """
        # item keys: (首轮)dict_keys(['observation', 'actions_left', 'suffix', 'system_prompt', 'task_idx', 'traj_reset_time', 'metrics', 'messages'])
    # item keys: dict_keys(['observation', 'actions_left', 'suffix', 'system_prompt', 'task_idx', 'traj_reset_time', 'metrics', 'messages', 'prompt_ids', 'response_ids', 'reward', 'penalty', 'llm_response'])
    # item keys: dict_keys(['observation', 'actions_left', 'suffix', 'messages', 'prompt_ids', 'response_ids', 'reward', 'penalty', 'llm_response', 'metrics'])
    # item keys: dict_keys(['observation', 'actions_left', 'suffix'])
        
        if not rollout_cache.history:
            print(f"[FORMULATE_ROLLOUTS] rollout_cache.history is empty! Creating default entry. stop_reason: {rollout_cache.stop_reason}, step: {rollout_cache.step}, rollout_cache: {rollout_cache}")
        # 补充当reach_max_turns和reach_max_length时history中信息缺失的情况
        item_keys = {'messages': list,'prompt_ids': list,'response_ids': list,'reward': float,'penalty': float,'llm_response': str,'metrics': dict}
        for item in rollout_cache.history:
            for key in item_keys:
                if key not in item:
                    item[key] = item_keys[key]()
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])

        # print(f"\n****** [FORMULATE_ROLLOUTS] history items start ***********\n")
        # for item in self.rollout_cache.history:
            # print(f"[FORMULATE_ROLLOUTS] history item keys: {item.keys()}")
        # print(f'last item: \n{self.rollout_cache.history[-1]}')
        # print(f"\n****** [FORMULATE_ROLLOUTS] history items end ***********")
        

        traj_messages = []
        for items in self.rollout_cache.history:
            if items.get("messages"): 
                traj_messages.extend(items.get("messages", []))
        
        # self.logger.info(f"\n\n[FORMULATE_ROLLOUTS]********** traj_messages **********")
        # for i, item in enumerate(traj_messages):
        #     self.logger.info(item)
        # self.logger.info(f'[FORMULATE_ROLLOUTS]metrics: {last_cache.get("metrics", {})}')
        # print(f"\n****** [FORMULATE_ROLLOUTS] traj_messages start ***********\n")
        # for item in traj_messages:
        #     print(item)
        # print(f"\n****** [FORMULATE_ROLLOUTS] traj_messages end ***********")

        last_cache.pop("reward", None)
        history.append(last_cache)

        # 补充奖励
        scores = [i["reward"] for i in self.rollout_cache.history]
        unittest_output, final_reward, info = self.env.calculate_reward(stop_reason=rollout_cache.stop_reason,mode=self.mode)
        scores[-1] = max(final_reward, max(scores))
        episode_score = scores[-1]

        # # # 训练集时，如果达到最大回合数或最大长度，则将奖励设置为0
        if self.mode == 'train' and (rollout_cache.stop_reason in ["reach_max_turn","reach_max_length"] or info.get("reach_max_turn",False) or info.get("reach_max_length",False)):
            episode_score = min(final_reward, 0.5)
            scores[-1] = episode_score
            unittest_output = f"{unittest_output}\n\nAttention: (Train Mode) Reach max turn or max length, episode_score will be set to 0/0.5. stop_reason: {rollout_cache.stop_reason}, final_reward: {final_reward}, episode_score: {episode_score}"
            self.logger.info(f'[FORMULATE_ROLLOUTS] Attention: (Train Mode) Reach max turn or max length, episode_score will be set to 0/0.5. stop_reason: {rollout_cache.stop_reason}, final_reward: {final_reward}, episode_score: {episode_score}')
            print(f"[FORMULATE_ROLLOUTS]⚠️ Attention: (Train Mode) Reach max turn or max length), episode_score will be set to 0/0.5, stop_reason: {rollout_cache.stop_reason}, final_reward: {final_reward}, episode_score: {episode_score}")

        token_ids = []
        prompt_masks = []
        response_masks = []
        step_response_length_list = []
        step_prompt_length_list = []
        for items in self.rollout_cache.history:
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))
            step_response_length = len(items["response_ids"])
            step_response_length_list.append(step_response_length)
            step_prompt_length = len(items["prompt_ids"])
            step_prompt_length_list.append(step_prompt_length)

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        step_response_length_tensor = torch.tensor(step_response_length_list, dtype=torch.float)
        step_prompt_length_tensor = torch.tensor(step_prompt_length_list, dtype=torch.float)

        prompt_masks_sys_obs = copy.deepcopy(prompt_masks)
        prompt_masks_sys_obs_tensor = torch.tensor(prompt_masks_sys_obs, dtype=torch.bool).unsqueeze(0)

        first_response_idx = response_masks.index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        # metrics = self.rollout_cache.history[-1].get("metrics", {})
        metrics = info.get("metrics", {})
        metrics['reward'] = episode_score

        # Compact Filtering: env_timeout, env_failed, max_length, truncated
        if metrics.get('env_timeout') or metrics.get('env_failed'):
            response_mask = torch.zeros_like(response_mask)
            prompt_mask = torch.zeros_like(prompt_mask)
            score_tensor = torch.zeros_like(score_tensor)
            print(f"[FORMULATE_ROLLOUTS]❌ Compact Filtering: env_timeout, env_failed, task_idx: {metrics.get('task_idx','')}")

        # close env
        print(
            f"[EnvManager]🟢closing env ... because of {self.rollout_cache.stop_reason}, task_idx: {metrics.get('task_idx','')}"
        )
        self.env.close(stop_reason=self.rollout_cache.stop_reason)
        self.logger.info(f"[FORMULATE_ROLLOUTS]🟢env closed .... because of {self.rollout_cache.stop_reason}, task_idx: {metrics.get('task_idx','')}")

        lm_input.batch.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "prompt_mask": prompt_mask,
                "scores": score_tensor,
            }
        )
        lm_input.non_tensor_batch.update(
            {
                "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                "tags": np.array([self.rollout_cache.tag], dtype=object),
                "frames": np.array([self.rollout_cache.frames], dtype=object),
                "step_scores": np.array([scores], dtype=object),
                "episode_scores": np.array([episode_score], dtype=object),
                "traj_rollout_time": np.array([float(metrics.get("traj_rollout_time") or 0)], dtype=object),
                "traj_env_time": np.array([float(metrics.get("traj_env_time") or 0)], dtype=object),
            }
        )

        # length
        avg_step_response_length = round(step_response_length_tensor.mean().item(), 2)
        avg_step_prompt_length = round(step_prompt_length_tensor.mean().item(), 2)
        max_step_response_length = round(step_response_length_tensor.max().item(), 2)
        max_step_prompt_length = round(step_prompt_length_tensor.max().item(), 2)
        min_step_response_length = round(step_response_length_tensor.min().item(), 2)
        min_step_prompt_length = round(step_prompt_length_tensor.min().item(), 2)

        # traj-level metric
        env_metric = {
            "success": float(metrics.get("success") if metrics.get("success") is not None else (episode_score > 0)),
            "reward": float(metrics.get("reward") if metrics.get("reward") is not None else episode_score),
            "truncated": float(metrics.get("truncated") or 0), 
            "env_timeout": float(metrics.get("env_timeout") or 0),  # 交互时间过长, 不计算reward
            "env_failed": float(metrics.get("env_failed") or 0),  # reset failed, 不计算reward
            "reach_max_length": (
                1.0 if rollout_cache.stop_reason in ["reach_max_length"] else 0.0
            ), 
            "reach_max_turn": (
                1.0 if rollout_cache.stop_reason in ["reach_max_turn"] else 0.0
            ), 
            "if_sandbox_failed": float(metrics.get("if_sandbox_failed") or 0), # 轨迹中出现sandbox异常次数
            "turn_count": int(metrics.get("turn_count") or 0),
            "retry_times": float(metrics.get("retry_times") or 0),
            "action_is_valid": float(metrics.get("action_is_valid") or 0),
            "action_is_effective": float(metrics.get("action_is_effective") or 0),
            "traj_reset_time": float(metrics.get("traj_reset_time") or 0),
            "traj_reward_time": float(metrics.get("traj_reward_time") or 0),
            "traj_env_time": float(metrics.get("traj_env_time") or 0),
            "traj_step_time": float(metrics.get("traj_step_time") or 0),
            "traj_step_time_avgturn": round(float(metrics.get("traj_step_time") or 0)/float(metrics.get("turn_count") or 1), 4),
            "traj_rollout_time": float(metrics.get("traj_rollout_time") or 0),
            "avg_step_response_length": avg_step_response_length,
            "avg_step_prompt_length": avg_step_prompt_length,
            "max_step_response_length": max_step_response_length,
            "max_step_prompt_length": max_step_prompt_length,
            "min_step_response_length": min_step_response_length,
            "min_step_prompt_length": min_step_prompt_length,
        }
        traj_keys = list(env_metric.keys())

        # step-level metric
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get("metrics", {}).items():
                if k in traj_keys:
                    continue
                if k == "task_idx":
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))
        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache.history)

        # add tag
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        # response_length
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}

        prompt_length = torch.tensor(prompt_masks_sys_obs).sum(dim=-1).float().mean().item()
        length = prompt_length + response_length
        max_seq_length = self.pipeline_config.sequence_length
        step_count = metrics.get("turn_count", (len(traj_messages)-2)//2)

        if metrics.get("env_timeout"):
            stop_reason = "env_timeout"
        elif metrics.get("env_failed"):
            stop_reason = "env_failed"
        elif metrics.get("reach_max_turn"):
            stop_reason = "reach_max_turn"
        elif metrics.get("truncated"):
            stop_reason = "truncated"
        elif rollout_cache.stop_reason in ["reach_max_length", "abort","reach_max_turn"]:
            stop_reason = rollout_cache.stop_reason
        elif rollout_cache.stop_reason == "self_submit" or metrics.get("success", True):
            stop_reason = "self_submit"
        else:
            stop_reason = "unknown"
        task_idx = metrics.get("task_idx", 0)

        # 保存重要信息
        save = {    
            "task_idx": task_idx,
            "env_id": self.env_config["env_id"],
            "group_id": self.env_config["group_id"],
            "tag": self.env_config["tag"],
            "length": length,
            "step_count": step_count,
            "stop_reason": stop_reason,
            "episode_score": episode_score,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "max_seq_length": max_seq_length,
            "traj_messages": traj_messages,
            "metrics": metrics,
            "env_metric": env_metric,
        }
        # time_str = time.strftime("%m%d%H%M%S", time.localtime())
        # log_path = os.path.join(self.pipeline_config.base_dir,'env_manager_traj',f'{self.current_step}-re{episode_score}_{stop_reason}_{tag}_{task_idx}_{time_str}-{self.env_config["env_id"]}_step{step_count}_rlgh{response_length}_plgh{prompt_length}_srlgh{avg_step_response_length}_splgh{avg_step_prompt_length}.json')
        # write_data_json(save,log_path)

        # 重要信息打印
        print(
            f"\n[formulate_rollouts]task_idx: {task_idx}, mode: {self.mode}" \
            f"[env_metric]({self.rollout_cache.tag}_group{self.rollout_cache.group_id}_seed{self.group_seed}_env{self.rollout_cache.env_id}){env_metric}" \
            f"[Token]length: {length}, max_seq_length: {max_seq_length}, prompt_length: {prompt_length}, response_length: {response_length}" \
            f"[stop_reason] {stop_reason} [step_count] {step_count} [episode_score] {episode_score}"
        )
        self.logger.info(
            f"[FORMULATE_ROLLOUTS]task_idx: {task_idx}, mode: {self.mode}" \
            f"[env_metric]({self.rollout_cache.tag}_group{self.rollout_cache.group_id}_seed{self.group_seed}_env{self.rollout_cache.env_id}){env_metric}" \
            f"[Token]length: {length}, max_seq_length: {max_seq_length}, prompt_length: {prompt_length}, response_length: {response_length}" \
            f"[stop_reason] {stop_reason} [step_count] {step_count} [episode_score] {episode_score}" \
            f"[unittest_output] {unittest_output}"
        )
        self.logger.info(f"\n\n[FORMULATE_ROLLOUTS]********** history **********\n{history}")
        self.logger.save()

        # colummns_config中的key在dump之后会从data_proto中移出
        lm_input.non_tensor_batch["model_name"] = np.array(
            [os.path.basename(self.pipeline_config.base_dir)], dtype=object
        )
        
        lm_input.non_tensor_batch["save_content"] = np.array([json.dumps(save)], dtype=object)
        lm_input.non_tensor_batch["step"] = np.array([self.current_step], dtype=object)
        lm_input.non_tensor_batch["task_idx"] = np.array([task_idx], dtype=object)
        lm_input.non_tensor_batch["stop_reason"] = np.array([stop_reason], dtype=object)
        lm_input.non_tensor_batch["mode"] = np.array([self.mode], dtype=object)
        lm_input.non_tensor_batch["episode_score"] = np.array([episode_score], dtype=object)
        
        # 检测并截断超长的unittest_output字段 (限制在8MB以内)
        max_unittest_output_bytes = 8 * 1024 * 1024  # 8MB
        unittest_output_bytes = len(unittest_output.encode('utf-8'))
        if unittest_output_bytes > max_unittest_output_bytes:
            truncated_bytes = max_unittest_output_bytes - 200
            truncated_output = unittest_output.encode('utf-8')[-truncated_bytes:].decode('utf-8', errors='ignore')
            unittest_output = f"{truncated_output}\n[WARNING] unittest_output truncated from {unittest_output_bytes} bytes to {len(unittest_output.encode('utf-8'))} bytes."
        
        lm_input.non_tensor_batch["unittest_output"] = np.array([unittest_output], dtype=object)
        colummns_config = [
            ["task_idx", "bigint"],
            ["model_name", "string"],
            ["stop_reason", "string"],
            ["episode_score", "double"],
            ["mode", "string"],
            ["save_content", "string"],
            ["unittest_output", "string"],
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        return lm_input
 