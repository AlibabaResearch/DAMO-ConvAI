import asyncio
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.utils.context_managers import local_profiler

from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.datasets.data_distributor import DataDistributorManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider, get_extra_data_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.checkpoint_manager import download_model
from roll.pipeline.agentic.env_manager.mcp_swe_env_manager import MCPSweEnvManager
from roll.utils.import_utils import safe_import_class

import socket
import random
def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)  # 设置超时时间
            result = s.connect_ex(('127.0.0.1', port))
            return result != 0  # 如果连接失败，说明端口可用
    except Exception:
        return True  # 如果出现异常，假设端口可用


class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None
        self.mode = "train"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(self,
                   pipeline_config,
                   generate_scheduler,
                   reward_scheduler,
                   output_queue,
                   collator: Optional[callable] = None,
                   mode: str = "train",
                   data_distributor: DataDistributorManager = None):
        super().initialize(pipeline_config)

        self.output_queue = output_queue
        self.mode = mode
        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(self.worker_config.model_args, model_name_or_path)
        self.processor = default_processor_provider(self.worker_config.model_args, model_name_or_path)
        self.reward_tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.reward.model_args)

        # 为每个env环境预分配端口，避免端口冲突
        allocated_ports = set()
        env_ports = {}
        for _env_id in self.env_configs.keys():
            attempts = 0
            max_attempts = 100  # 最大尝试次数
            # 生成随机端口号
            start_port = random.randint(3000, 9999)
            while attempts < max_attempts:
                # 检查端口是否已被分配且可用
                if start_port not in allocated_ports and is_port_available(start_port):
                    allocated_ports.add(start_port)
                    env_ports[_env_id] = start_port
                    break
                start_port += 1
                start_port = start_port % 7000 + 3000
                attempts += 1

            # 如果达到最大尝试次数仍未找到可用端口，使用最后一个尝试的端口
            if attempts >= max_attempts:
                start_port = random.randint(3000, 9999)
                allocated_ports.add(start_port)
                env_ports[_env_id] = start_port

        def create_env_manager(env_id, env_config):
            if env_id == 0:
                self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
            env_manager_cls = safe_import_class(env_config["env_manager_cls"])

            assert env_manager_cls is not None
            tokenizer = copy.deepcopy(self.tokenizer)
            processor = copy.deepcopy(self.processor)
            reward_tokenizer = copy.deepcopy(self.reward_tokenizer)
            extra_data_provider = None
            if processor is not None and isinstance(processor, ProcessorMixin):
                extra_data_provider = get_extra_data_provider(model_name_or_path, processor=processor)
            if env_manager_cls == MCPSweEnvManager:
                return env_id, env_manager_cls(
                    worker_config=self.worker_config,
                    pipeline_config=pipeline_config,
                    env_config=env_config,
                    tokenizer=tokenizer,  # https://github.com/huggingface/tokenizers/issues/537
                    reward_tokenizer=reward_tokenizer,
                    generate_scheduler=generate_scheduler,
                    reward_scheduler=reward_scheduler,
                    output_queue=output_queue,
                    thread_lock=self.thread_lock,
                    mode=mode,
                    data_distributor=data_distributor,
                    env_port=env_ports[env_id]
                )
            else:
                return env_id, env_manager_cls(
                    worker_config=self.worker_config,
                    pipeline_config=pipeline_config,
                    env_config=env_config,
                    tokenizer=tokenizer,  # https://github.com/huggingface/tokenizers/issues/537
                    processor=processor,
                    generate_scheduler=generate_scheduler,
                    output_queue=output_queue,
                    thread_lock=self.thread_lock,
                    mode=mode,
                    extra_data_provider=extra_data_provider,
                )
        with ThreadPoolExecutor(max_workers=min(len(self.env_configs), 64)) as executor:
            futures = [
                executor.submit(create_env_manager, env_id, env_config)
                for env_id, env_config in self.env_configs.items()
            ]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    self.logger.error(f"Failed to initialize env_manager: {e}", exc_info=True)
                    raise e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, seed):
        # Set environment variables for profiler context
        os.environ["roll_EXEC_FUNC_NAME"] = "run_rollout_loop"
        os.environ["WORKER_NAME"] = f"EnvironmentWorker_{self.rank}"
        
        loop = asyncio.get_event_loop()
        pool = ThreadPoolExecutor(max_workers=len(self.env_managers))
        
        def run_with_profiler(env_manager, data_proto):
            with local_profiler():
                return env_manager.run_rollout_loop(data_proto)
        
        def run_without_profiler(env_manager, data_proto):
            return env_manager.run_rollout_loop(data_proto)
        
        tasks = []
        for env_id, env_manager in self.env_managers.items():
            # Only profile the first env_manager (env_id=0) on rank=0
            run_func = run_without_profiler
            if self.rank == 0 and env_id == 0:
                run_func = run_with_profiler
            tasks.append(loop.run_in_executor(pool, run_func, env_manager, DataProto(meta_info={"seed": seed})))
        
        await asyncio.gather(*tasks)
        pool.shutdown()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step):
        for env_manager in self.env_managers.values():
            env_manager.update_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        for env_manager in self.env_managers.values():
            env_manager.stop()
