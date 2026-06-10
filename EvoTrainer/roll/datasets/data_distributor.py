import json
import random
import time
import os
import hashlib
import tqdm
from ray.runtime_env import RuntimeEnv
from roll.utils.constants import RAY_NAMESPACE
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import ray
from ray.util.queue import Queue
from roll.utils.logging import get_logger
from roll.distributed.scheduler.protocol import DataProto

logger = get_logger()


def create_data_shards(data_list: List[Any], group_ids: List[int], shard_strategy: str = "hash") -> Dict[
    int, List[Any]]:
    """
    为每个group_id创建数据分片

    Args:
        data_list: 原始数据列表
        group_ids: group_id列表(经过去重的列表)
        shard_strategy: 分片策略，支持"hash"和"round_robin"

    Returns:
        Dict[int, List[Any]]: {group_id: shard_data_list}
    """
    if not data_list or not group_ids:
        return {}

    shards = {}
    total_data_count = len(data_list)
    group_count = len(group_ids)

    if shard_strategy == "hash":
        # 基于hash的分片策略
        for group_id in group_ids:
            shard_data = []
            for i in range(total_data_count):
                # 使用group_id和索引计算hash
                hash_input = f"{group_id}_{i}"
                hash_value = hashlib.md5(hash_input.encode()).hexdigest()
                hash_int = int(hash_value, 16)
                data_index = hash_int % total_data_count
                shard_data.append(data_list[data_index])
            shards[group_id] = shard_data

    elif shard_strategy == "round_robin":
        # 轮询分片策略
        for i, group_id in enumerate(group_ids):
            shard_data = []
            for j in range(i, total_data_count, group_count):
                shard_data.append(data_list[j])
            shards[group_id] = shard_data

    else:
        # 默认策略：每个group_id获得完整数据副本
        for group_id in group_ids:
            shards[group_id] = data_list.copy()

    return shards


@ray.remote
class DataDistributor:
    """
    数据分发Actor，负责为不同env tag加载和分发数据
    支持基于数据分片的并行化数据分发策略
    每个env tag的每个group_id都有独立的数据分片
    支持成功轨迹缓存功能，为每个训练环境建立缓存池
    """

    def __init__(self, env_data_path: Dict[str, str], distributor_name: str = None,
                 shard_strategy: str = "round_robin"):
        """
        Args:
            env_data_path: {env_tag: data_file_path} 配置每个env tag对应的数据文件路径
            distributor_name: 分发器名称，用于标识不同的分发器实例
            shard_strategy: 数据分片策略，支持"hash"、"round_robin"和"full_copy"
        """
        # 设置分发器名称和环境变量
        self.distributor_name = distributor_name
        self.shard_strategy = shard_strategy

        self.env_data_path = env_data_path  # 记录每个env的data path
        self.original_data: Dict[str, List[Any]] = {}  # 原始数据缓存
        self.data_shards: Dict[str, Dict[int, List[Any]]] = {}  # {env_tag: {group_id: shard_data}}
        self.shard_tracker: Dict[
            str, Dict[int, int]] = {}  # {env_tag: {group_id: max_shard_idx}} 跟踪每个group_id的最大shard_idx

        # 成功轨迹缓存相关属性
        self.success_trajectory_cache: Dict[str, List[Dict[
            str, Any]]] = {}  # {env_tag + native_id: [{'trajectory_data': DataProto, 'rollout_global_step': int, 'timestamp': float}]}
        self.last_cache_cleanup = time.time()  # 上次缓存清理时间
        self.cache_cleanup_interval = 300  # 缓存清理间隔（秒）
        self.current_global_step = 0  # 当前全局步数

        # 添加健康检查相关属性
        self.last_heartbeat = time.time()
        self.is_healthy = True
        self.error_count = 0
        self.max_errors = 10  # 最大错误次数，超过后标记为不健康

        # 初始化每个env tag的数据结构
        for env_tag in env_data_path.keys():
            self.original_data[env_tag] = []
            self.data_shards[env_tag] = {}
            self.shard_tracker[env_tag] = {}

        logger.info(
            f"DataDistributor initialized with name: {self.distributor_name}, shard_strategy: {self.shard_strategy}")

    def _update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = time.time()

    def _check_health(self) -> bool:
        """检查Actor健康状态"""
        current_time = time.time()
        # 如果超过5分钟没有心跳更新，认为不健康
        if current_time - self.last_heartbeat > 300:
            self.is_healthy = False
            logger.warning(
                f"DataDistributor: Health check failed - no heartbeat for {current_time - self.last_heartbeat:.2f} seconds")

        return self.is_healthy

    def _handle_error(self, error: Exception):
        """处理错误，更新错误计数"""
        self.error_count += 1
        logger.error(f"DataDistributor: Error occurred (count: {self.error_count}/{self.max_errors}) - {error}")

        if self.error_count >= self.max_errors:
            self.is_healthy = False
            logger.error(f"DataDistributor: Too many errors ({self.error_count}), marking as unhealthy")

    def heartbeat(self) -> bool:
        """心跳检查，用于监控Actor状态"""
        try:
            self._update_heartbeat()
            health_status = self._check_health()
            logger.debug(f"DataDistributor: Heartbeat - healthy: {health_status}, error_count: {self.error_count}")
            return health_status
        except Exception as e:
            logger.error(f"DataDistributor: Heartbeat failed - {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取Actor状态信息"""
        return {
            "healthy": self.is_healthy,
            "error_count": self.error_count,
            "last_heartbeat": self.last_heartbeat,
            "env_count": len(self.env_data_path),
            "data_shards_count": len(self.data_shards),
            "shard_tracker_count": len(self.shard_tracker)
        }

    def load_data_for_env(self, env_tag: str):
        """为指定env(tag)加载数据"""
        try:
            self._update_heartbeat()

            if env_tag not in self.env_data_path:
                logger.warning(f"DataDistributor: No data config found for env_tag: {env_tag}")
                return

            file_path = self.env_data_path[env_tag]

            # 如果文件路径为空，说明这个环境不需要加载数据，只需要跟踪shard_idx
            if not file_path:
                logger.info(
                    f"DataDistributor: No data file path for env_tag: {env_tag}, skipping data loading but will track shard_idx")
                self.original_data[env_tag] = []
                return

            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

            data_list = []
            with open(file_path, 'r', encoding='utf-8') as fr:
                line_num = 0
                for line in tqdm.tqdm(fr, desc=f"Loading data for env_tag: {env_tag} from path: {file_path}"):
                    line_num += 1
                    line = line.strip()
                    if line:
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"DataDistributor: Invalid JSON at line {line_num} in {file_path}: {e}")
                            continue

            # random.shuffle(data_list)  # Commented out to preserve difficulty-based sorting
            self.original_data[env_tag] = data_list
            logger.info(
                f"DataDistributor: Loaded data - env_tag={env_tag}, file_path={file_path}, data_count={len(data_list)}")

        except Exception as e:
            self._handle_error(e)
            logger.error(
                f"DataDistributor: Failed to load data for env_tag={env_tag}, file_path={file_path}, error={e}")
            self.original_data[env_tag] = []

    def create_shards_for_env(self, env_tag: str, env_group_mapping: Dict[str, List[int]]):
        """为指定env创建数据分片"""
        try:
            self._update_heartbeat()

            group_ids = env_group_mapping[env_tag]

            # 这里需要对group_id进行去重，否则数据分片会出问题。
            unique_group_ids = list(set(group_ids))

            # 检查是否有数据需要分片
            if env_tag not in self.original_data or not self.original_data[env_tag]:
                logger.info(
                    f"DataDistributor: No data available for creating shards - env_tag={env_tag}, but will initialize shard_tracker")
                # 即使没有数据，也要初始化shard_tracker
                self.data_shards[env_tag] = {}
                for group_id in unique_group_ids:
                    self.shard_tracker[env_tag][group_id] = 0
                return

            data_list = self.original_data[env_tag]

            # 创建数据分片
            shards = create_data_shards(data_list, unique_group_ids, self.shard_strategy)
            self.data_shards[env_tag] = shards

            logger.info(
                f"DataDistributor: Created shards - env_tag={env_tag}, group_count={len(group_ids)}, shard_strategy={self.shard_strategy}")

            for group_id, shard_data in shards.items():
                logger.info(f"DataDistributor: Group {group_id} shard size: {len(shard_data)}")

            # 初始化shard_tracker
            for group_id in unique_group_ids:
                self.shard_tracker[env_tag][group_id] = 0

        except Exception as e:
            self._handle_error(e)
            logger.error(f"DataDistributor: Failed to create shards for env_tag={env_tag}, error={e}")

    def get_data_for_worker(self, env_tag: str, group_id: int, shard_idx: int) -> Optional[Any]:
        """为指定worker获取一条数据
        env_tag: 环境tag
        group_id: 环境组id
        shard_idx: shard索引，环境组内的shard索引
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, returning None")
                return None

            # 检查是否有数据分片
            if env_tag not in self.data_shards or group_id not in self.data_shards[env_tag]:
                logger.warning(f"DataDistributor: No shard data available - env_tag={env_tag}, group_id={group_id}")
                return None

            shard_data = self.data_shards[env_tag][group_id]
            if not shard_data:
                logger.info(
                    f"DataDistributor: No data available for this environment - env_tag={env_tag}, group_id={group_id}")
                return None

            data_index = shard_idx % len(shard_data)
            data = shard_data[data_index]
            return data

        except Exception as e:
            self._handle_error(e)
            logger.error(
                f"DataDistributor: Failed to get data for worker - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}, error={e}")
            return None

    def get_next_shard_idx(self, env_tag: str, group_id: int) -> int:
        """获取指定env_tag和group_id的下一个shard_idx

        Args:
            env_tag: 环境tag
            group_id: 环境组id

        Returns:
            int: 下一个shard_idx (当前最大shard_idx + 1)
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, returning 0")
                return 0

            if env_tag not in self.shard_tracker or group_id not in self.shard_tracker[env_tag]:
                logger.warning(f"DataDistributor: No shard tracker found - env_tag={env_tag}, group_id={group_id}")
                return 0

            latest_shard_idx = self.shard_tracker[env_tag][group_id]
            next_shard_idx = latest_shard_idx + 1
            return next_shard_idx

        except Exception as e:
            self._handle_error(e)
            logger.error(
                f"DataDistributor: Failed to get next shard_idx - env_tag={env_tag}, group_id={group_id}, error={e}")

            return 0

    def sync_shard_idx(self, env_tag: str, group_id: int, shard_idx: int):
        """同步指定env_tag和group_id的shard_idx

        Args:
            env_tag: 环境tag
            group_id: 环境组id
            shard_idx: 最新的shard_idx
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, skipping sync")
                return

            current_max_shard_idx = self.shard_tracker.get(env_tag, {}).get(group_id, 0)

            if shard_idx > current_max_shard_idx:
                self.shard_tracker[env_tag][group_id] = shard_idx

        except Exception as e:
            self._handle_error(e)
            logger.error(
                f"DataDistributor: Failed to sync shard_idx - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}, error={e}")

    def update_global_step(self, global_step: int):
        """更新当前全局步数，用于缓存清理

        Args:
            global_step: 当前全局步数
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, skipping global step update")
                return

            self.current_global_step = global_step
            logger.info(f"DataDistributor: Updated global step to {global_step}")

        except Exception as e:
            self._handle_error(e)
            logger.error(f"DataDistributor: Failed to update global step - global_step={global_step}, error={e}")

    def store_success_trajectory(self, env_tag: str, native_id: str, trajectory_data: DataProto,
                                 rollout_global_step: int) -> bool:
        """将成功轨迹存入缓存池

        Args:
            env_tag: 环境tag
            native_id: 数据本身的native_id
            trajectory_data: 轨迹数据（DataProto类型）
            rollout_global_step: 采样轨迹时的全局步数

        Returns:
            bool: 是否成功存入缓存
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, skipping trajectory storage")
                return False

            # 生成缓存key
            cache_key = f"{env_tag}_{native_id}"

            # 存储轨迹数据
            cache_data = {
                "rollout_global_step": rollout_global_step,
                "trajectory_data": trajectory_data,
                "timestamp": time.time()
            }

            # 如果key不存在，创建新的轨迹列表
            if cache_key not in self.success_trajectory_cache:
                self.success_trajectory_cache[cache_key] = []

            # 将新的轨迹数据添加到列表中
            self.success_trajectory_cache[cache_key].append(cache_data)

            logger.info(f"DataDistributor: Stored success trajectory - env_tag={env_tag}, native_id={native_id}, "
                        f"global_step={rollout_global_step}, trajectory_count={len(self.success_trajectory_cache[cache_key])}, "
                        f"trajectory_length={len(trajectory_data)}")

            # 检查是否需要清理缓存
            self._check_and_cleanup_cache()

            return True

        except Exception as e:
            self._handle_error(e)
            logger.error(f"DataDistributor: Failed to store success trajectory - env_tag={env_tag}, "
                         f"native_id={native_id}, global_step={rollout_global_step}, error={e}")
            return False

    def get_success_trajectory(self, env_tag: str, native_id: str) -> Optional[List[Dict[str, Any]]]:
        """从缓存池取出成功轨迹

        Args:
            env_tag: 环境tag
            native_id: 数据本身的native_id

        Returns:
            Optional[List[Dict[str, Any]]]: 轨迹数据列表，每个元素包含rollout_global_step和trajectory_data（DataProto类型），如果不存在则返回None
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, cannot get trajectory")
                return None

            # 生成缓存key
            cache_key = f"{env_tag}_{native_id}"

            if cache_key in self.success_trajectory_cache:
                trajectory_list = self.success_trajectory_cache[cache_key]
                logger.info(f"DataDistributor: Retrieved success trajectories - env_tag={env_tag}, "
                            f"native_id={native_id}, trajectory_count={len(trajectory_list)}")
                return trajectory_list
            else:
                logger.debug(
                    f"DataDistributor: No trajectory found in cache - env_tag={env_tag}, native_id={native_id}")
                return None

        except Exception as e:
            self._handle_error(e)
            logger.error(f"DataDistributor: Failed to get success trajectory - env_tag={env_tag}, "
                         f"native_id={native_id}, error={e}")
            return None

    def _check_and_cleanup_cache(self):
        """检查并清理缓存，只保留上一个global_step的数据"""
        current_time = time.time()

        # 检查是否需要执行清理
        if current_time - self.last_cache_cleanup < self.cache_cleanup_interval:
            return

        self.last_cache_cleanup = current_time

        try:
            keys_to_remove = []
            total_removed_trajectories = 0

            for cache_key, trajectory_list in self.success_trajectory_cache.items():
                # 过滤出不过期的轨迹
                valid_trajectories = []
                removed_count = 0

                for trajectory_data in trajectory_list:
                    stored_global_step = trajectory_data.get("rollout_global_step", 0)

                    # 如果存储的global_step比当前global_step小2或更多，则删除
                    if self.current_global_step - stored_global_step >= 2:
                        removed_count += 1
                    else:
                        valid_trajectories.append(trajectory_data)

                # 更新轨迹列表
                if removed_count > 0:
                    self.success_trajectory_cache[cache_key] = valid_trajectories
                    total_removed_trajectories += removed_count
                    logger.debug(
                        f"DataDistributor: Cleaned up {removed_count} expired trajectories from cache_key={cache_key}")

                # 如果轨迹列表为空，标记为删除
                if not valid_trajectories:
                    keys_to_remove.append(cache_key)

            # 删除空的缓存key
            for key in keys_to_remove:
                del self.success_trajectory_cache[key]
                logger.info(f"DataDistributor: Removed empty cache key={key}")

            if total_removed_trajectories > 0:
                logger.info(
                    f"DataDistributor: Cleaned up {total_removed_trajectories} expired trajectories from {len(keys_to_remove)} empty cache keys")

        except Exception as e:
            logger.error(f"DataDistributor: Error during cache cleanup - {e}")

    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态信息

        Returns:
            Dict[str, Any]: 缓存状态信息
        """
        try:
            self._update_heartbeat()

            if not self.is_healthy:
                logger.warning(f"DataDistributor: Actor is unhealthy, cannot get cache status")
                return {}

            cache_status = {
                "current_global_step": self.current_global_step,
                "last_cache_cleanup": self.last_cache_cleanup,
                "cache_cleanup_interval": self.cache_cleanup_interval,
                "total_cache_keys": len(self.success_trajectory_cache),
                "cache_details": {}
            }

            total_trajectories = 0
            for cache_key, trajectory_list in self.success_trajectory_cache.items():
                trajectory_count = len(trajectory_list)
                total_trajectories += trajectory_count
                cache_status["cache_details"][cache_key] = {
                    "trajectory_count": trajectory_count,
                    "latest_global_step": max(
                        [t.get("rollout_global_step", 0) for t in trajectory_list]) if trajectory_list else 0
                }

            cache_status["total_trajectories"] = total_trajectories

            return cache_status

        except Exception as e:
            self._handle_error(e)
            logger.error(f"DataDistributor: Failed to get cache status - error={e}")
            return {}


class DataDistributorManager:
    """
    数据分发管理器，负责创建和管理DataDistributor, train和val的数据分发器互相隔离
    支持基于数据分片的并行化数据分发策略
    """

    def __init__(self, env_configs: Dict = None, mode: str = "train", shard_strategy: str = "round_robin"):
        """
        Args:
            env_configs: 环境配置字典，包含每个env的data_path配置
            mode: 模式，用于区分训练("train")和验证("val")，默认为"train"
            shard_strategy: 数据分片策略，支持"hash"、"round_robin"和"full_copy"
        """
        self.env_configs = env_configs or {}
        self.mode = mode
        self.shard_strategy = shard_strategy
        self.env_data_path = {}
        self.distributor: Optional[ray.actor.ActorHandle] = None
        self.placement_group = None  # 添加placement group引用
        self.env_group_mapping: Dict[str, List[int]] = defaultdict(
            list)  # {env_tag: [group_ids]} 记录每个env tag对应的group id
        self.reverse_env_group_mapping: Dict[int, str] = {}  # {group_id: env_tag} 记录每个group id对应的env tag

        # 添加监控和恢复相关属性
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 300秒检查一次健康状态
        self.restart_count = 0
        self.max_restarts = 5  # 最大重启次数
        self.last_restart_time = 0
        self.restart_cooldown = 300  # 重启冷却时间（秒）

        # 从custom_envs中提取data_path配置
        for worker_rank in self.env_configs:
            worker_rank_env_configs = self.env_configs[worker_rank]
            for idx, env_config in worker_rank_env_configs.items():
                original_env_tag = env_config['tag']
                group_id = env_config['group_id']

                # 为环境名称添加模式后缀, 方便在日志中记录
                env_tag = f"{original_env_tag}_{self.mode}"

                # 记录所有环境，无论是否有数据路径
                self.env_group_mapping[env_tag].append(group_id)
                self.reverse_env_group_mapping[group_id] = original_env_tag

                # 检查是否有数据路径配置
                if self.mode == 'train' and hasattr(env_config['config'], 'train_data_path'):
                    data_path = env_config['config'].train_data_path
                    self.env_data_path[env_tag] = data_path
                elif self.mode == 'val' and hasattr(env_config['config'], 'eval_data_path'):
                    data_path = env_config['config'].eval_data_path
                    self.env_data_path[env_tag] = data_path

    def _should_restart(self) -> bool:
        """判断是否应该重启DataDistributor"""
        current_time = time.time()

        # 检查重启次数限制
        if self.restart_count >= self.max_restarts:
            logger.error(f"DataDistributorManager: Max restart count ({self.max_restarts}) reached")
            return False

        # 检查重启冷却时间
        if current_time - self.last_restart_time < self.restart_cooldown:
            logger.warning(f"DataDistributorManager: Restart cooldown not finished yet")
            return False

        return True

    def _perform_health_check(self) -> bool:
        """执行健康检查"""
        if not self.distributor:
            return False

        try:
            import ray
            # 检查Actor是否还活着
            status = ray.get(self.distributor.get_status.remote(), timeout=10.0)
            is_healthy = status.get("healthy", False)

            if not is_healthy:
                logger.warning(f"DataDistributorManager: DataDistributor is unhealthy - {status}")
                return False

            # 执行心跳检查
            heartbeat_ok = ray.get(self.distributor.heartbeat.remote(), timeout=5.0)
            if not heartbeat_ok:
                logger.warning(f"DataDistributorManager: DataDistributor heartbeat failed")
                return False

            logger.debug(f"DataDistributorManager: Health check passed - {status}")
            return True

        except Exception as e:
            logger.error(f"DataDistributorManager: Health check failed - {e}")
            return False

    def _monitor_and_recover(self):
        """监控并自动恢复DataDistributor"""
        current_time = time.time()

        # 检查是否需要执行健康检查
        if current_time - self.last_health_check < self.health_check_interval:
            return

        self.last_health_check = current_time

        # 执行健康检查
        if not self._perform_health_check():
            logger.warning(f"DataDistributorManager: Health check failed, attempting recovery...")
            if self._should_restart():
                self.restart_distributor()
            else:
                logger.error(f"DataDistributorManager: Cannot restart DataDistributor due to limits")

    def start(self):
        """启动数据分发器"""
        # 检查是否有环境需要跟踪shard_idx（有env_group_mapping就说明有环境需要跟踪）
        if not self.env_group_mapping:
            logger.warning(
                f"DataDistributorManager: No environments found for mode '{self.mode}', DataDistributor will not be started")
            return

        # 如果没有数据路径但有环境需要跟踪shard_idx，创建一个空的env_data_path
        if not self.env_data_path:
            logger.info(
                f"DataDistributorManager: No data_path configuration found for mode '{self.mode}', but will start DataDistributor for shard_idx tracking")
            # 为所有环境创建一个空的data_path，这样distributor可以启动但不加载数据
            for env_tag in self.env_group_mapping.keys():
                self.env_data_path[env_tag] = ""

        # 为每个模式创建不同的分发器名称
        distributor_name = f"DataDistributor_{self.mode}"

        # 创建运行时环境，设置环境变量
        env_vars = {
            "CLUSTER_NAME": "DataDistributor",
            "RANK": "0",
            "WORKER_NAME": distributor_name,
            "ROLL_LOG_DIR": os.environ.get("ROLL_LOG_DIR", "./output/logs"),
        }
        runtime_env = RuntimeEnv(env_vars=env_vars)

        try:
            # 创建placement group用于在主节点部署DataDistributor
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            # 创建placement group，只需要CPU资源
            pg = placement_group([{"CPU": 1}], strategy="STRICT_PACK")
            ray.get(pg.ready())

            # 保存placement group引用
            self.placement_group = pg

            logger.info(f"DataDistributorManager: Created placement group for DataDistributor on master node")

            # 使用PlacementGroupSchedulingStrategy确保在主节点部署
            self.distributor = DataDistributor.options(
                name=distributor_name,
                namespace=RAY_NAMESPACE,
                runtime_env=runtime_env,
                num_cpus=1,  # 1 CPU资源
                max_restarts=3,  # 允许Actor自动重启
                max_task_retries=3,  # 允许任务重试
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0  # 使用第一个bundle
                )
            ).remote(self.env_data_path, distributor_name, self.shard_strategy)

            # 等待Actor启动
            logger.info(f"DataDistributorManager: Waiting for DataDistributor to start on master node...")
            time.sleep(1)

            # 为每个env tag加载初始数据
            for env_tag in self.env_data_path.keys():
                try:
                    logger.info(f"DataDistributorManager: Loading data for env_tag={env_tag}")
                    ray.get(self.distributor.load_data_for_env.remote(env_tag), timeout=600.0)  # 增加超时时间
                except ray.exceptions.GetTimeoutError:
                    logger.error(f"DataDistributorManager: Timeout loading data for env_tag={env_tag}")
                    raise
                except Exception as e:
                    logger.error(f"DataDistributorManager: Error loading data for env_tag={env_tag}, error={e}")
                    raise

            # 为每个env tag创建数据分片,并初始化shard_tracker
            for env_tag in self.env_data_path.keys():
                try:
                    logger.info(f"DataDistributorManager: Creating shards for env_tag={env_tag}")
                    ray.get(self.distributor.create_shards_for_env.remote(env_tag, self.env_group_mapping),
                            timeout=600.0)
                except ray.exceptions.GetTimeoutError:
                    logger.error(f"DataDistributorManager: Timeout creating shards for env_tag={env_tag}")
                    raise
                except Exception as e:
                    logger.error(f"DataDistributorManager: Error creating shards for env_tag={env_tag}, error={e}")
                    raise

            # 执行初始健康检查
            if not self._perform_health_check():
                raise RuntimeError("DataDistributor failed initial health check")

            logger.info(
                f"DataDistributorManager: Started DataDistributor '{distributor_name}' for mode '{self.mode}' with config: {self.env_data_path}, shard_strategy: {self.shard_strategy}")

        except Exception as e:
            logger.error(f"DataDistributorManager: Failed to start DataDistributor - {e}")
            self.distributor = None
            raise

    def get_data_for_worker(self, env_tag: str, group_id: int, shard_idx: int = None) -> Optional[Any]:
        """为worker获取数据
        env_tag: 环境tag（原始tag，会自动添加模式后缀）
        group_id: 环境组id
        shard_idx: shard索引，如果提供则使用基于shard_idx的数据分发策略
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            # 为环境名称添加模式后缀
            env_tag_with_suffix = f"{env_tag}_{self.mode}"

            try:
                # 添加超时机制，防止无限等待
                import ray
                data = ray.get(
                    self.distributor.get_data_for_worker.remote(env_tag_with_suffix, group_id, shard_idx),
                    timeout=30.0  # 30秒超时
                )
                logger.info(
                    f"DataDistributorManager: Success Sending data - env_tag={env_tag}, mode={self.mode}, group_id={group_id}, shard_idx={shard_idx}")
                return data
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    f"DataDistributorManager: Timeout getting data - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return None
            except Exception as e:
                logger.error(
                    f"DataDistributorManager: Error getting data - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}, error={e}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return None
        else:
            logger.warning(
                f"DataDistributorManager: Distributor not started, cannot get data - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}")

            return None

    def get_next_shard_idx(self, env_tag: str, group_id: int) -> int:
        """获取指定env_tag和group_id的下一个shard_idx

        Args:
            env_tag: 环境tag（原始tag，会自动添加模式后缀）
            group_id: 环境组id

        Returns:
            int: 下一个shard_idx (当前最大shard_idx + 1)
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            # 为环境名称添加模式后缀
            env_tag_with_suffix = f"{env_tag}_{self.mode}"
            try:
                # 添加超时机制，防止无限等待
                import ray
                next_shard_idx = ray.get(
                    self.distributor.get_next_shard_idx.remote(env_tag_with_suffix, group_id),
                    timeout=5.0  # 5秒超时
                )
                logger.info(
                    f"DataDistributorManager: Requesting next shard_idx - env_tag={env_tag}, mode={self.mode}, group_id={group_id}")
                return next_shard_idx
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    f"DataDistributorManager: Timeout getting next shard_idx - env_tag={env_tag}, group_id={group_id}, mode={self.mode}")

                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return 0
            except Exception as e:
                logger.error(
                    f"DataDistributorManager: Error getting next shard_idx - env_tag={env_tag}, group_id={group_id}, mode={self.mode}, error={e}")

                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return 0
        else:
            logger.warning(
                f"DataDistributorManager: Distributor not started, cannot get next shard_idx - env_tag={env_tag}, group_id={group_id}, mode={self.mode}")
            return 0

    def sync_shard_idx(self, group_id: int, shard_idx: int):
        """同步指定env_tag和group_id的shard_idx

        Args:
            env_tag: 环境tag（原始tag，会自动添加模式后缀）
            group_id: 环境组id
            shard_idx: 最新的shard_idx
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            env_tag = self.reverse_env_group_mapping[group_id]
            env_tag_with_suffix = f"{env_tag}_{self.mode}"

            try:
                # 添加超时机制，防止无限等待
                import ray
                ray.get(
                    self.distributor.sync_shard_idx.remote(env_tag_with_suffix, group_id, shard_idx),
                    timeout=30.0  # 30秒超时
                )
                logger.info(
                    f"DataDistributorManager: Synced shard_idx - env_tag={env_tag}, group_id={group_id}, shard_idx={shard_idx}")
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    f"DataDistributorManager: Timeout syncing shard_idx - group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}")
            except Exception as e:
                logger.error(
                    f"DataDistributorManager: Error syncing shard_idx - group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}, error={e}")

        else:
            logger.warning(
                f"DataDistributorManager: Distributor not started, cannot sync shard_idx - group_id={group_id}, shard_idx={shard_idx}, mode={self.mode}")
            return

    def check_distributor_status(self) -> bool:
        """检查DataDistributor Actor的状态

        Returns:
            bool: True表示Actor正常，False表示Actor异常或未启动
        """
        if not self.distributor:
            logger.warning("DataDistributorManager: Distributor not initialized")
            return False

        try:
            import ray
            # 尝试调用一个简单的方法来检查Actor是否还活着
            ray.get(self.distributor.get_next_shard_idx.remote("test_env", 0), timeout=5.0)
            return True
        except Exception as e:
            logger.error(f"DataDistributorManager: Distributor status check failed - {e}")
            return False

    def restart_distributor(self):
        """重启DataDistributor Actor"""
        current_time = time.time()

        if not self._should_restart():
            logger.warning("DataDistributorManager: Cannot restart due to limits")
            return

        logger.info("DataDistributorManager: Restarting DataDistributor...")

        try:
            # 清理旧的Actor
            if self.distributor:
                try:
                    import ray
                    ray.kill(self.distributor)
                except:
                    pass
                self.distributor = None

            # 清理旧的placement group
            if self.placement_group:
                try:
                    import ray
                    from ray.util.placement_group import remove_placement_group
                    remove_placement_group(self.placement_group)
                except:
                    pass
                self.placement_group = None

            # 重置重启计数
            self.restart_count += 1
            self.last_restart_time = current_time

            # 重新启动
            self.start()

            if self.check_distributor_status():
                logger.info(
                    f"DataDistributorManager: DataDistributor restarted successfully (attempt {self.restart_count}/{self.max_restarts})")
            else:
                logger.error(
                    f"DataDistributorManager: Failed to restart DataDistributor (attempt {self.restart_count}/{self.max_restarts})")

        except Exception as e:
            logger.error(f"DataDistributorManager: Error during restart - {e}")
            self.distributor = None

    def update_global_step(self, global_step: int):
        """更新当前全局步数，用于缓存清理

        Args:
            global_step: 当前全局步数
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            try:
                # 添加超时机制，防止无限等待
                import ray
                ray.get(
                    self.distributor.update_global_step.remote(global_step),
                    timeout=5.0  # 5秒超时
                )
                logger.info(f"DataDistributorManager: Updated global step to {global_step}")
            except ray.exceptions.GetTimeoutError:
                logger.error(f"DataDistributorManager: Timeout updating global step - global_step={global_step}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
            except Exception as e:
                logger.error(
                    f"DataDistributorManager: Error updating global step - global_step={global_step}, error={e}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
        else:
            logger.warning(
                f"DataDistributorManager: Distributor not started, cannot update global step - global_step={global_step}")

    def store_success_trajectory(self, env_tag: str, native_id: str, trajectory_data: DataProto,
                                 global_step: int) -> bool:
        """将成功轨迹存入缓存池

        Args:
            env_tag: 环境tag（原始tag，会自动添加模式后缀）
            native_id: 数据本身的native_id
            trajectory_data: 轨迹数据（DataProto类型）
            global_step: 当前全局步数

        Returns:
            bool: 是否成功存入缓存
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            try:
                # 添加超时机制，防止无限等待
                import ray
                success = ray.get(
                    self.distributor.store_success_trajectory.remote(env_tag, native_id, trajectory_data, global_step),
                    timeout=30.0  # 30秒超时
                )
                if success:
                    logger.info(f"DataDistributorManager: Successfully stored trajectory - env_tag={env_tag}, "
                                f"native_id={native_id}, global_step={global_step}, trajectory_length={len(trajectory_data)}")
                else:
                    logger.warning(f"DataDistributorManager: Failed to store trajectory - env_tag={env_tag}, "
                                   f"native_id={native_id}, global_step={global_step}")
                return success
            except ray.exceptions.GetTimeoutError:
                logger.error(f"DataDistributorManager: Timeout storing trajectory - env_tag={env_tag}, "
                             f"native_id={native_id}, global_step={global_step}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return False
            except Exception as e:
                logger.error(f"DataDistributorManager: Error storing trajectory - env_tag={env_tag}, "
                             f"native_id={native_id}, global_step={global_step}, error={e}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return False
        else:
            logger.warning(f"DataDistributorManager: Distributor not started, cannot store trajectory - "
                           f"env_tag={env_tag}, native_id={native_id}, global_step={global_step}")
            return False

    def get_success_trajectory(self, env_tag: str, native_id: str) -> Optional[List[Dict[str, Any]]]:
        """从缓存池取出成功轨迹

        Args:
            env_tag: 环境tag（原始tag，会自动添加模式后缀）
            native_id: 数据本身的native_id

        Returns:
            Optional[List[Dict[str, Any]]]: 轨迹数据列表，每个元素包含rollout_global_step和trajectory_data（DataProto类型），如果不存在则返回None
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            try:
                # 添加超时机制，防止无限等待
                import ray
                trajectory_list = ray.get(
                    self.distributor.get_success_trajectory.remote(env_tag, native_id),
                    timeout=30.0  # 30秒超时
                )
                if trajectory_list:
                    logger.info(f"DataDistributorManager: Retrieved trajectories from cache - env_tag={env_tag}, "
                                f"native_id={native_id}, trajectory_count={len(trajectory_list)}")
                else:
                    logger.debug(
                        f"DataDistributorManager: No trajectory found in cache - env_tag={env_tag}, native_id={native_id}")
                return trajectory_list
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    f"DataDistributorManager: Timeout getting trajectory - env_tag={env_tag}, native_id={native_id}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return None
            except Exception as e:
                logger.error(f"DataDistributorManager: Error getting trajectory - env_tag={env_tag}, "
                             f"native_id={native_id}, error={e}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return None
        else:
            logger.warning(f"DataDistributorManager: Distributor not started, cannot get trajectory - "
                           f"env_tag={env_tag}, native_id={native_id}")
            return None

    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态信息

        Returns:
            Dict[str, Any]: 缓存状态信息
        """
        # 执行监控和恢复
        self._monitor_and_recover()

        if self.distributor:
            try:
                # 添加超时机制，防止无限等待
                import ray
                cache_status = ray.get(
                    self.distributor.get_cache_status.remote(),
                    timeout=10.0  # 10秒超时
                )
                logger.debug(f"DataDistributorManager: Retrieved cache status - {cache_status}")
                return cache_status
            except ray.exceptions.GetTimeoutError:
                logger.error(f"DataDistributorManager: Timeout getting cache status")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return {}
            except Exception as e:
                logger.error(f"DataDistributorManager: Error getting cache status - error={e}")
                # 尝试重启DataDistributor
                if self._should_restart():
                    self.restart_distributor()
                return {}
        else:
            logger.warning(f"DataDistributorManager: Distributor not started, cannot get cache status")
            return {}

