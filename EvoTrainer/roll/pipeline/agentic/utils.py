import copy
import multiprocessing
import json
import os
import os.path
import shutil
import subprocess
from datetime import datetime
from multiprocessing import Pool
from typing import List, Callable, Dict, Optional

import imageio
import numpy as np
import torch
from codetiming import Timer
from torch import Tensor
import json

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import AgenticConfig, RewardNormalizationConfig
from roll.pipeline.rlvr.utils import DUMPING_FUNC
from roll.utils.logging import get_logger
from roll.utils.functionals import (
    masked_whiten,
    compute_gae_advantage_return,
    compute_clip_fraction,
    compute_reinforce_return,
    compute_approx_kl,
)

logger = get_logger()

def save_eval_batch_info(save_dir: str, step: int, eval_batch, env_ids: List, tags: List):
    """
    保存评估批次的重要信息

    Args:
        save_dir: 保存目录
        step: 当前步骤
        eval_batch: 评估批次数据
        env_ids: 环境ID列表
        tags: 环境标签列表
    """
    with Timer(name="save_eval_info", logger=None) as timer:
        try:
            # 创建保存目录
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            eval_save_dir = os.path.join(save_dir, f"eval_step_{step}_{timestamp}")
            os.makedirs(eval_save_dir, exist_ok=True)

            # 提取评估批次中的重要信息
            eval_info_list = []

            # 获取基本信息
            raw_messages_list = []
            for term in eval_batch.non_tensor_batch["messages_list"].tolist():
                raw_messages_list.append(term.tolist())
            final_rewards = eval_batch.non_tensor_batch["final_reward"].tolist()
            detailed_reward_info = eval_batch.non_tensor_batch['detailed_reward_info'].tolist()
            judge_info = eval_batch.non_tensor_batch['judge_info'].tolist()
            meta_data = eval_batch.non_tensor_batch["meta_data"].tolist()

            # 为每个样本创建详细信息
            for i, (env_id, tag) in enumerate(zip(env_ids, tags)):
                eval_info = {
                    "env_id": env_id,
                    "tag": tag,
                    "final_reward": final_rewards[i],
                    "meta_data": meta_data[i],
                    "detailed_reward_info": detailed_reward_info[i],
                    "judge_info": judge_info[i],
                    "messages": raw_messages_list[i],
                }

                eval_info_list.append(eval_info)

            # 保存详细信息到JSON文件
            detailed_info_file = os.path.join(eval_save_dir, "detailed_eval_info.json")
            with open(detailed_info_file, 'w', encoding='utf-8') as f:
                json.dump(eval_info_list, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved eval batch info to {eval_save_dir}/detailed_eval_info.json")
            logger.info(f"Total samples: {len(eval_info_list)}")

        except Exception as e:
            logger.error(f"Failed to save eval batch info: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print('eval_info_list:', eval_info_list)

        logger.info(f"save_eval_batch_info_cost: {timer.last}")


def dump_rollout_render(save_dir, step, frames: List[List], env_ids: List, tags: List, episode_scores: List):
    with Timer(name="dump", logger=None) as timer:
        try:
            local_save_dir = f'/tmp/rollout_render/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            os.makedirs(local_save_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            args_list = [
                (os.path.join(local_save_dir, f"{step}", f"{env_id}_{tag}_{episode_score:.1f}.gif"), frame_list)
                for frame_list, env_id, tag, episode_score in zip(frames, env_ids, tags, episode_scores)
                if len(frame_list) > 0
            ]
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with Pool(processes=16) as pool:
                pool.starmap(dump_frames_as_gif, args_list)

            rar_file_path = os.path.join(
                "/tmp", f'rollout_render_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{step}.zip'
            )
            command = ["zip", "-rq", rar_file_path, local_save_dir]
            subprocess.run(command, check=True)
            shutil.move(rar_file_path, save_dir)
            shutil.rmtree(local_save_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"dump rollout render failed: {e}")
    logger.info(f"dump_rollout_render_cost: {timer.last}")


@torch.no_grad()
def compute_discounted_returns(batch: DataProto, adv_estimator, gamma=1.0) -> DataProto:
    """
    Compute discounted returns for each trajectory in the batch.

    Args:
        batch (DataProto): A `DataProto` instance containing trajectories.
        adv_estimator (str): Advantage estimator type; only `"gigpo"` triggers computation here.
        gamma (float, optional): Discount factor applied to future rewards. Defaults to 1.0.

    Returns:
        DataProto: Updated batch where each trajectory contains an extra tensor key
                   `"step_rewards"` holding the computed discounted returns.
    """
    if adv_estimator in ["gigpo", "step_reinforce"]:
        batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
        batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
        for traj_id, traj_batch in batch_group_by_traj.items():

            indices: Tensor = torch.argsort(torch.from_numpy(traj_batch.non_tensor_batch["step"].astype(np.int64)))
            traj_batch.reorder(indices)
            step_scores = traj_batch.non_tensor_batch["step_scores"].astype(np.float32)
            rewards = torch.as_tensor(step_scores).float()
            discounts = torch.empty_like(rewards)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return
            traj_batch.batch["step_rewards"] = discounts

        merged = DataProto.concat(list(batch_group_by_traj.values()))
        merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
        merged.pop("sample_order_placeholder")
        return merged
    else:
        return batch

# TODO: 这里的功能性和rlvr比较接近，但因为后续agentic会有潜在的修改需求，所以就先拎出来
@torch.no_grad()
def agentic_reward_norm(batch: "DataProto", reward_normalization: RewardNormalizationConfig, pipeline_config: AgenticConfig) -> torch.Tensor:
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    grouping = reward_normalization.grouping
    norm_mean_type = reward_normalization.norm_mean_type
    norm_std_type = reward_normalization.norm_std_type

    all_scores = batch.batch["scores"].float()
    batch_mean = None
    batch_std = None
    if norm_mean_type == "batch":
        batch_mean = all_scores.mean()
    if norm_std_type == "batch":
        batch_std = all_scores.std()

    batch_list = []
    batch_grouped: Dict[str, DataProto] = {"default": batch}
    if grouping != "batch":
        batch_grouped = batch.group_by(keys=grouping)
    for group_name, group_batch in batch_grouped.items():
        # sequence_length = group_batch.non_tensor_batch["sequence_length"]
        dialogue_turns = group_batch.non_tensor_batch["dialogue_turns"]
        scores_with_bonus = group_batch.batch["scores"].sum(dim=-1)
        # response_mask = group_batch.batch["response_mask"]
        if pipeline_config.add_length_bonus:
            # 添加基于对话轮次的奖励分数
            # 对于得分最高的轨迹（score==1），对话轮次最少的获得额外0.2分
            max_score_mask = (scores_with_bonus == 1)
            if max_score_mask.any():
                # 计算得分等于1的轨迹的平均对话轮次
                avg_turns = dialogue_turns[max_score_mask].mean()
                # 在得分最高的样本中找到对话轮次最少的
                max_score_dialogue_turns = dialogue_turns[max_score_mask]
                min_turns = max_score_dialogue_turns.min()
                # 对话轮次必须小于平均值
                if min_turns < avg_turns:
                    # 在得分最高且对话轮次最少的样本上添加奖励
                    bonus_mask = max_score_mask & (dialogue_turns == min_turns)
                    scores_with_bonus += bonus_mask.float() * 0.2

        if not pipeline_config.add_token_penalty:
            original_dtype = scores_with_bonus.dtype
            scores_float = scores_with_bonus.float()

            if norm_mean_type == "batch":
                reward_mean = batch_mean
            elif norm_mean_type == "group":
                reward_mean = scores_float.mean()
            else:
                reward_mean = 0.0

            if norm_std_type == "batch":
                reward_std = batch_std
            elif norm_std_type == "group":
                reward_std = scores_float.std()
            else:
                reward_std = None

            if reward_std is not None:
                # 处理单个元素或标准差为0的情况，避免除以0
                if scores_float.numel() > 1 and reward_std.abs() > 1e-6:
                    normalized_scores = (scores_float - reward_mean) / (reward_std + 1e-6)
                else:
                    normalized_scores = torch.zeros_like(scores_float)
            else:
                normalized_scores = scores_float - reward_mean

            normalized_scores = normalized_scores.to(dtype=original_dtype)
            group_batch.batch["grouped_rewards"] = normalized_scores
        else:
            group_batch.batch["grouped_rewards"] = scores_with_bonus
        batch_list.append(group_batch)

    batch = DataProto.concat(batch_list)
    batch.reorder(indices=torch.argsort(batch.batch["sample_order_placeholder"]))
    batch.pop("sample_order_placeholder")
    return batch.batch.pop("grouped_rewards")


def build_state_group(batch: "DataProto") -> "DataProto":
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    batch_group_by_traj_group: Dict[str, DataProto] = batch.group_by(keys="traj_group_id")
    merged = []
    for traj_group_id, traj_group_batch in batch_group_by_traj_group.items():
        batch_group_by_state: Dict[str, DataProto] = traj_group_batch.group_by(keys="state_hash")
        for state, state_batch in batch_group_by_state.items():
            state_batch.non_tensor_batch["state_group_id"] = np.array(
                [state] * state_batch.batch.batch_size[0], dtype=object
            )
            merged.append(state_batch)
    state_batch_size = [len(m) for m in merged]
    merged = DataProto.concat(merged)
    merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
    merged.pop("sample_order_placeholder")
    metrics = merged.meta_info.pop("metrics", {})
    metrics["system/state_batch_size/max"] = np.max(state_batch_size)
    metrics["system/state_batch_size/mean"] = np.mean(state_batch_size)
    metrics["system/state_batch_size/min"] = np.min(state_batch_size)
    merged.meta_info["metrics"] = metrics
    return merged


@torch.no_grad()
def compute_response_level_rewards(batch: "DataProto", pipeline_config: AgenticConfig) -> "DataProto":
    reward_metrics = {}
    if pipeline_config.adv_estimator == "gigpo":
        # ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/gigpo/core_gigpo.py#L109
        # step 1
        episode_scores = torch.from_numpy(batch.non_tensor_batch["episode_scores"].astype(np.float32))
        scores_to_group = DataProto.from_dict({"scores": episode_scores})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        episode_rewards: torch.Tensor = agentic_reward_norm(
            scores_to_group, reward_normalization=pipeline_config.reward_normalization
        )

        # step 2
        batch = build_state_group(batch=batch)

        # step 3
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        step_rewards: torch.Tensor = agentic_reward_norm(
            batch=scores_to_group,
            reward_normalization=RewardNormalizationConfig(
                grouping="state_group_id", method=pipeline_config.reward_normalization.method
            ),
        )

        batch.batch["response_level_rewards"] = (
            pipeline_config.episode_reward_weight * episode_rewards + pipeline_config.step_reward_weight * step_rewards
        )
        batch.batch["episode_rewards_norm"] = episode_rewards
        batch.batch["step_rewards_norm"] = step_rewards
    elif pipeline_config.adv_estimator == "step_reinforce":
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        batch.batch["response_level_rewards"] = agentic_reward_norm(
            scores_to_group, reward_normalization=pipeline_config.reward_normalization
        )
    else:
        batch.batch["response_level_rewards"] = agentic_reward_norm(
            batch, reward_normalization=pipeline_config.reward_normalization, pipeline_config=pipeline_config
        )

    # 加上clip
    if pipeline_config.reward_clip:
        reward_metrics["critic/reward_clip_frac"] = compute_clip_fraction(
            values=batch.batch["response_level_rewards"],
            clip_min=-pipeline_config.reward_clip,
            clip_max=pipeline_config.reward_clip,
        )
        batch.batch["response_level_rewards"] = torch.clamp(
            batch.batch["response_level_rewards"], min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    return batch, reward_metrics

@torch.no_grad()
def get_agentic_sample_level_mask(data: DataProto, pipeline_config: AgenticConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"].clone()

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    # 1. 过滤异常结束的轨迹（超长也会被赋值成-999）
    scores = data.batch["scores"].clone().sum(dim=-1)
    valid_mask = (scores >= -5) & (scores <= 5) # 留下正常轨迹
    final_sample_mask = final_sample_mask * valid_mask.float()

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["response_mask"] = final_response_mask
    return data, mask_metrics

@torch.no_grad()
def get_agentic_response_level_mask(data: "DataProto", pipeline_config: AgenticConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    if getattr(pipeline_config, "max_len_mask", False):
        # TODO 当前是混合多个的action/state，需要去判别，或者用别的方式过滤
        final_sample_mask = final_sample_mask
        mask_metrics["actor/max_len_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


print_only_once = False


def dump_frames_as_gif(filename, frames, duration=0.2):
    global print_only_once
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with imageio.get_writer(filename, mode="v", duration=duration) as writer:
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))

    except Exception as e:
        if not print_only_once:
            print(f"Error saving gif: {e}")
        print_only_once = True
        pass


def remove_nan_items(data: Dict[str, np.ndarray]):
    if not data:
        return {}

    # 所有数组都假设 dtype=object，只有 None 需要过滤
    arr = np.vstack([np.asarray(v, dtype=object) for v in data.values()])  # (num_keys, N)
    mask = arr != None  # noqa: E711
    valid_row_mask = mask.all(axis=0)
    return {
        k: np.asarray(v, dtype=object)[valid_row_mask]
        for k, v in data.items()
    }


def dump_rollout_trajectories(path, global_step, data: DataProto):
    """
    Dumps rollout trajectories to persistent storage.

    The data is written using a column-based configuration defined in COLUMMNS_CONFIG.
    Each column is specified as a list [column_name, data_type], where:
    - column_name: string identifier for the column
    - data_type: data type specification ('bigint', 'string', 'double', etc.)

    Example configuration:
    colummns_config = [
        ['global_step', 'bigint'],
        ['id', 'string'],
        ['source', 'string'],
        # ... additional columns
    ]
    """
    if not path:
        return

    columns_config: Optional[List] = data.meta_info.get("COLUMMNS_CONFIG", None)
    if columns_config is None:
        return

    write_data = {item[0]: data.non_tensor_batch.pop(item[0]) for item in columns_config if item[0] in data.non_tensor_batch}

    write_data = remove_nan_items(copy.deepcopy(write_data))
    data_cnt = len(write_data[columns_config[0][0]])

    write_data["global_step"] = [global_step] * data_cnt
    columns_config.append(["global_step", "bigint"])

    for checker, func in DUMPING_FUNC:
        if checker(path):
            p = multiprocessing.Process(target=func, args=(path, write_data, columns_config), daemon=False)
            p.start()


def compute_segment_masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    对每段连续的1分别计算 masked_mean，不连续的段不相乘。

    Args:
        tensor: [batch_size, seq_len] 要计算的值
        mask: [batch_size, seq_len] mask，1表示有效位置，0表示无效位置

    Returns:
        [batch_size, seq_len] 结果，每段连续的1位置填充该段的 masked_mean
    """
    batch_size, seq_len = mask.shape
    device = mask.device
    result = torch.zeros_like(tensor)

    # 对每个样本分别处理
    for b in range(batch_size):
        sample_mask = mask[b]  # [seq_len]
        sample_tensor = tensor[b]  # [seq_len]

        # 找到所有连续的1的段
        # 使用 diff 找到边界：1->0 和 0->1 的位置
        diff = torch.diff(sample_mask, prepend=torch.tensor([0], device=device))
        # 找到段的开始位置（0->1）
        segment_starts = torch.where(diff == 1)[0]
        # 找到段的结束位置（1->0），diff[i]==-1 表示 mask[i-1]==1 且 mask[i]==0，所以段的结束位置是 i（不包括i）
        segment_ends = torch.where(diff == -1)[0]

        # 如果最后一个位置是1，需要添加结束位置
        if sample_mask[-1] == 1:
            segment_ends = torch.cat([segment_ends, torch.tensor([seq_len], device=device)])

        # 确保 segment_starts 和 segment_ends 长度匹配
        if len(segment_starts) != len(segment_ends):
            # 如果长度不匹配，只处理能匹配的部分
            min_len = min(len(segment_starts), len(segment_ends))
            segment_starts = segment_starts[:min_len]
            segment_ends = segment_ends[:min_len]

        # 对每段分别计算 masked_mean
        for start, end in zip(segment_starts, segment_ends):
            # 获取这段的索引
            segment_indices = torch.arange(start, end, device=device)
            segment_mask = sample_mask[segment_indices]  # 这段的mask
            segment_tensor = sample_tensor[segment_indices]  # 这段的值

            if segment_mask.sum() > 0:
                # 计算这段的 masked_mean（只考虑mask为1的位置）
                segment_mean = (segment_tensor * segment_mask).sum() / (segment_mask.sum() + 1e-8)
                # 将结果填充到这段内mask为1的位置
                result[b, segment_indices] = segment_mean * segment_mask

    return result


def compute_agentic_reinforce_return(
    token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor, mask: Optional[torch.Tensor] = None
):
    """
    计算 REINFORCE 的 return，支持按 mask 分段 discount 衰减。
    每段内所有位置获得相同的折扣累积值（从该段最后位置开始累积）。

    Args:
        token_level_rewards: [batch_size, seq_len] token 级别的奖励
        gamma: discount factor
        lambd: lambda 参数（当前未使用，保留以兼容接口）
        mask: [batch_size, seq_len] mask，1表示有效位置，0表示无效位置。如果为None，则对所有位置计算

    Returns:
        advantages: [batch_size, seq_len] advantages
        returns: [batch_size, seq_len] returns
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        device = token_level_rewards.device
        returns = torch.zeros_like(token_level_rewards, dtype=torch.float32)

        # 如果没有提供 mask，则对所有位置计算（向后兼容）
        if mask is None:
            mask = torch.ones_like(token_level_rewards)

        # 确保 gamma 是标量
        gamma_val = gamma.item() if torch.is_tensor(gamma) else gamma

        # 对每个样本分别处理
        for b in range(batch_size):
            sample_mask = mask[b]  # [seq_len]
            sample_rewards = token_level_rewards[b]  # [seq_len]

            # 找到所有连续的1的段
            # 使用 diff 找到边界：1->0 和 0->1 的位置
            diff = torch.diff(sample_mask.float(), prepend=torch.tensor([0.0], device=device))

            # 找到段的开始位置（0->1，diff==1）
            segment_starts = torch.where(diff == 1)[0]

            # 找到段的结束位置（1->0，diff==-1）
            segment_ends = torch.where(diff == -1)[0]

            # 如果最后一个位置是1，需要添加结束位置
            if len(sample_mask) > 0 and sample_mask[-1] == 1:
                segment_ends = torch.cat([segment_ends, torch.tensor([gen_len], device=device)])

            # 计算该段从最后位置开始的累积折扣奖励
            cumulative_return = 0.0
            # 对每段分别计算 discounted return
            for start, end in zip(segment_starts.flip(-1), segment_ends.flip(-1)):
                start_idx = start.item()
                end_idx = end.item()
                segment_len = end_idx - start_idx

                cumulative_return = sample_rewards[end_idx - 1].item() + gamma_val * cumulative_return

                # 该段内所有位置都设置为这个累积值
                returns[b, start_idx:end_idx] = cumulative_return

        advantages = returns

    return advantages, returns


@torch.no_grad()
def agentic_compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
    pipeline_config=None,
    divide_std=None,
    add_token_penalty=None
):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    if response_mask.sum() == 0:
        whiten_rewards = False
        whiten_advantages = False
        logger.info("Warning: domain final_response_mask.sum() == 0! All masked_whiten will be skipped.")

    # Check OPD config
    is_pure_opd = getattr(pipeline_config, "is_pure_opd", False) if pipeline_config else False
    use_opd = getattr(pipeline_config, "use_opd", False) if pipeline_config else False
    opd_kl_coef = getattr(pipeline_config, "opd_kl_coef", 1.0) if pipeline_config else 1.0

    # Compute KL divergence for OPD modes
    kld = None
    if is_pure_opd or use_opd:
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"] if getattr(pipeline_config, "enable_old_logprobs_recompute", False) else data.batch["infer_logprobs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=getattr(pipeline_config, "kl_penalty", "kl"),
        )

    # For pure OPD mode, advantage is directly -kld
    if is_pure_opd:
        advantages = -kld
        returns = advantages
        data.batch["raw_advantages"] = advantages
    else:
        token_level_rewards = data.batch["token_level_rewards"].float()
        if whiten_rewards:
            token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
        token_level_rewards = token_level_rewards * response_mask
        data.batch["token_level_rewards"] = token_level_rewards
        if adv_estimator == "gae":
            values = data.batch["values"].float()
            data.batch["values"] = values * response_mask
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
            )
        elif adv_estimator in ["reinforce", "grpo", "gigpo", "step_reinforce"]:
            advantages, returns = compute_reinforce_return(
                token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
            )
        elif adv_estimator in ["agentic_reinforce"]:
            advantages, returns = compute_agentic_reinforce_return(
                token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd, mask=response_mask
            )
        else:
            raise NotImplementedError

    # token level reward 赋值
    # 到这一步为止'advantages'其实还是reward
    if add_token_penalty:
        token_penalty = data.batch["token_penalty"][:, 1:] # select the response part
        # 如果token_penalty小于0，使用token_penalty的值；否则使用advantages的值
        # advantages = torch.where(token_penalty < 0, token_penalty, advantages)
        # advantages += token_penalty
        advantages = token_penalty
    data.batch["raw_advantages"] = advantages

    # Apply mixed OPD mode
    if use_opd:
        advantages = advantages - opd_kl_coef * kld

    if pipeline_config.adv_estimator == "grpo":
        # 在token粒度计算grpo中的组内advantage
        advantages = grouped_whiten_advantages(batch=data, reward_normalization=pipeline_config.reward_normalization,
                                               divide_std=divide_std)

    if whiten_advantages:
        # TODO whiten过程中是否要考虑response的长度？
        advantages = masked_whiten(values=advantages, mask=response_mask, divide_std=divide_std)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

def grouped_whiten_advantages(batch: "DataProto", reward_normalization, divide_std:bool) -> torch.Tensor:
    from roll.distributed.scheduler.protocol import DataProto
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    grouping = reward_normalization.grouping
    batch_grouped: Dict[str, DataProto] = {"default": batch}
    if grouping != "batch":
        batch_grouped = batch.group_by(keys=grouping)
    batch_list = []
    for group_name, group_batch in batch_grouped.items():
        batch_advantages = group_batch.batch["raw_advantages"]
        batch_response_mask = group_batch.batch["response_mask"][:, 1:]
        if batch_response_mask.sum() == 0:
            pass
        else:
            batch_advantages = masked_whiten(values=batch_advantages, mask=batch_response_mask, divide_std=divide_std)
        group_batch.batch["advantages"] = batch_advantages
        batch_list.append(group_batch)
    batch = DataProto.concat(batch_list)
    batch.reorder(indices=torch.argsort(batch.batch["sample_order_placeholder"]))
    batch.pop("sample_order_placeholder")
    return batch.batch.pop("advantages")

@torch.no_grad()
def get_score_normalize_fn(rn_cfg) -> Callable:
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if method == "mean_std":
        return normalize_mean_std
    elif method == "mean":
        return normalize_mean
        # norm_func: Callable[..., Any] = lambda x: (x - x.mean(dim=-1, keepdim=True))
    elif method == "asym_clip":
        return normalize_asym_clip
    elif method == "identity":
        norm_func = lambda x: x
    else:
        raise ValueError(f"Invalid normalization method: {method}")

    return norm_func

def normalize_mean_std(x):
    """使用均值标准差归一化，排除异常轨迹（reward在正负5范围之外）"""
    # 创建有效数据的掩码，排除异常轨迹（在正负5范围之外）
    valid_mask = (x >= -5) & (x <= 5)

    # 如果没有有效数据，返回零张量
    if not valid_mask.any():
        return torch.zeros_like(x)

    valid_x = x[valid_mask]
    # 只使用有效数据计算均值和标准差
    if valid_x.numel() > 1:
        mean_val = valid_x.mean(dim=-1, keepdim=True)
        std_val = valid_x.std(dim=-1, keepdim=True)

        std_val_max = valid_x.std(dim=-1, keepdim=True).abs().max()
        # 如果标准差太小，使用零张量
        if std_val_max <= 1e-6:
            return torch.zeros_like(x)
    else:
        return torch.zeros_like(x)

    # 将异常值（在正负5范围之外）替换为计算出的均值
    x_processed = x.clone()
    x_processed[~valid_mask] = mean_val

    # 对所有数据进行归一化
    normalized = (x_processed - mean_val) / (std_val + 1e-6)

    # 检查是否出现nan或inf
    nan_mask = torch.isnan(normalized)
    inf_mask = torch.isinf(normalized)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()

    if nan_count > 0 or inf_count > 0:
        logger.info(f"***** DATA ERROR ***** Detected {nan_count} nan and {inf_count} inf samples")
        logger.info(f"original data: {x.tolist()}")
        logger.info(f"normalized data: {normalized.tolist()}")

        # 将nan和inf位置替换为0
        normalized = torch.where(torch.isnan(normalized) | torch.isinf(normalized),
                               torch.zeros_like(normalized), normalized)

        logger.info(f"***** REPAIR ERROR DATA ***** Replaced {nan_count + inf_count} samples with 0")

    return normalized


def normalize_mean(x):
    """使用均值标准差归一化，排除异常轨迹（reward在正负5范围之外）"""
    # 创建有效数据的掩码，排除异常轨迹（在正负5范围之外）
    valid_mask = (x >= -5) & (x <= 5)

    # 如果没有有效数据，返回零张量
    if not valid_mask.any():
        return torch.zeros_like(x)

    valid_x = x[valid_mask]
    # 只使用有效数据计算均值和标准差
    mean_val = valid_x.mean(dim=-1, keepdim=True)

    # 将异常值（在正负5范围之外）替换为计算出的均值
    x_processed = x.clone()
    x_processed[~valid_mask] = mean_val

    # 对所有数据进行归一化
    normalized = x_processed - mean_val

    # 检查是否出现nan或inf
    nan_mask = torch.isnan(normalized)
    inf_mask = torch.isinf(normalized)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()

    if nan_count > 0 or inf_count > 0:
        logger.info(f"***** DATA ERROR ***** Detected {nan_count} nan and {inf_count} inf samples")
        logger.info(f"original data: {x.tolist()}")
        logger.info(f"normalized data: {normalized.tolist()}")

        # 将nan和inf位置替换为0
        normalized = torch.where(torch.isnan(normalized) | torch.isinf(normalized),
                               torch.zeros_like(normalized), normalized)

        logger.info(f"***** REPAIR ERROR DATA ***** Replaced {nan_count + inf_count} samples with 0")

    return normalized

def normalize_asym_clip(x):
    """使用均值标准差归一化，排除异常轨迹（reward在正负5范围之外）"""
    # 创建有效数据的掩码，排除异常轨迹（在正负5范围之外）
    valid_mask = (x >= -5) & (x <= 5)

    # 如果没有有效数据，返回零张量
    if not valid_mask.any():
        return torch.zeros_like(x)

    valid_x = x[valid_mask]
    # 只使用有效数据计算均值和标准差
    if valid_x.numel() > 1:
        mean_val = valid_x.mean(dim=-1, keepdim=True)
        std_val = valid_x.std(dim=-1, keepdim=True)

        std_val_max = valid_x.std(dim=-1, keepdim=True).abs().max()
        # 如果标准差太小，使用零张量
        if std_val_max <= 1e-6:
            return torch.zeros_like(x)
    else:
        return torch.zeros_like(x)

    # 将异常值（在正负5范围之外）替换为计算出的均值
    x_processed = x.clone()
    x_processed[~valid_mask] = mean_val

    # 对所有数据进行归一化
    normalized = (x_processed - mean_val) / (std_val + 1e-6)

    # 对有效数据进行非对称裁剪
    normalized = normalized.clamp(min=-1, max=3)


    # 检查是否出现nan或inf
    nan_mask = torch.isnan(normalized)
    inf_mask = torch.isinf(normalized)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()

    if nan_count > 0 or inf_count > 0:
        logger.info(f"***** DATA ERROR ***** Detected {nan_count} nan and {inf_count} inf samples")
        logger.info(f"original data: {x.tolist()}")
        logger.info(f"normalized data: {normalized.tolist()}")

        # 将nan和inf位置替换为0
        normalized = torch.where(torch.isnan(normalized) | torch.isinf(normalized),
                               torch.zeros_like(normalized), normalized)

        logger.info(f"***** REPAIR ERROR DATA ***** Replaced {nan_count + inf_count} samples with 0")

    return normalized