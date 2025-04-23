import asyncio
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, List, Set, cast

import gin
from absl import flags
from absl.flags import FLAGS
from rich.logging import RichHandler
from tqdm import tqdm

from sotopia.agents.llm_agent import LLMAgent
from sotopia.database.env_agent_combo_storage import (
    EnvAgentComboStorage,
)
from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages.message_classes import AgentAction, Observation
from sotopia.samplers.base_sampler import BaseSampler, EnvAgentCombo
from sotopia.server import run_async_server
from sotopia_conf.gin_utils import parse_gin_flags, run
from sotopia_conf.server import _DEFAULT_GIN_SEARCH_PATHS

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
    ],
)


# get all episode logs
def get_all_episodes() -> List[EpisodeLog]:
    episode_pks: List[str] = list(EpisodeLog.all_pks())
    all_episodes = []
    for pk in tqdm(episode_pks):
        try:
            curr_ep = EpisodeLog.get(pk)
        except Exception as _:
            continue
        all_episodes.append(curr_ep)
    print(f"all episodes loaded {len(all_episodes)}")
    return all_episodes


# all env-agent combos
def get_all_env_agent_combos(
    start_combo_idx: int, end_combo_idx: int
) -> Dict[str, EnvAgentComboStorage]:
    experiment_env_pks = list(EnvironmentProfile.all_pks())
    all_combos_map: Dict[str, EnvAgentComboStorage] = {}

    for env_pk in experiment_env_pks:
        env_agent_combo_storage_list = list(
            EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_pk).all()
        )[start_combo_idx:end_combo_idx]
        for combo in env_agent_combo_storage_list:
            all_combos_map[cast(str, combo.pk)] = cast(EnvAgentComboStorage, combo)
    print(f"all combos loaded {len(all_combos_map)}")
    return all_combos_map


def find_combo_pk(
    env_pk: str,
    agent_id1: str,
    agent_id2: str,
    all_combos_map: Dict[str, EnvAgentComboStorage],
) -> str | None:
    for combo_key in all_combos_map:
        combo = all_combos_map[combo_key]
        curr_tuple = (combo.env_id, combo.agent_ids[0], combo.agent_ids[1])
        if curr_tuple == (env_pk, agent_id1, agent_id2):
            return combo_key
    return None


def get_combo_model_map(
    all_episodes: List[EpisodeLog],
    all_combos_map: Dict[str, EnvAgentComboStorage],
) -> Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]]:
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]] = (
        defaultdict(Counter)
    )
    bad_combos = []
    valid_count = 0
    invalid_count = 0
    bad_rewards_count = 0
    bad_gpt4_rewards_count = 0
    bad_combo_count = 0

    # iterate through episodes
    for i in tqdm(range(len(all_episodes))):
        curr_ep = all_episodes[i]
        bad_rewards = False

        # check if episode is valid
        if not curr_ep.models:
            invalid_count += 1
            continue

        # check if rewards are valid
        for idx, model in enumerate(curr_ep.models[1:]):
            if not isinstance(curr_ep.rewards[idx], tuple):
                bad_rewards = True
                break
        if bad_rewards:
            bad_rewards_count += 1
            if tuple(curr_ep.models) == ("gpt4", "gpt4", "gpt4"):
                bad_gpt4_rewards_count += 1
            continue

        # find combo pk by env pk and agent ids
        curr_combo_pk = find_combo_pk(
            curr_ep.environment,
            curr_ep.agents[0],
            curr_ep.agents[1],
            all_combos_map,
        )
        if curr_combo_pk:
            model_pair: tuple[LLM_Name, LLM_Name, LLM_Name] = cast(
                tuple[LLM_Name, LLM_Name, LLM_Name], tuple(curr_ep.models)
            )
            combo_model_map[curr_combo_pk][model_pair] += 1
            valid_count += 1
        else:
            bad_combos.append(
                (curr_ep.environment, curr_ep.agents[0], curr_ep.agents[1])
            )

            bad_combo_count += 1
    print("-" * 20 + "Episode Parsing Summary" + "-" * 20)
    print(f"valid episodes: {valid_count}")
    print(f"invalid episodes (missing episode.models): {invalid_count}")
    print(f"bad combo: {bad_combo_count}")
    print(f"bad rewards: {bad_rewards_count}")
    print(f"bad gpt4 rewards: {bad_gpt4_rewards_count}")
    return combo_model_map


def get_all_model_pairs(
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]],
) -> Set[tuple[LLM_Name, LLM_Name, LLM_Name]]:
    all_model_pairs = set()
    for key in combo_model_map:
        for combo in combo_model_map[key]:
            all_model_pairs.add(combo)

    # print all model pairs
    print("-" * 20 + "All Model Pairs" + "-" * 20)
    for pair in all_model_pairs:
        print(pair)
    print()
    return all_model_pairs


def get_all_missing_model_pairs(
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]],
    all_model_pairs: Set[tuple[LLM_Name, LLM_Name, LLM_Name]],
    num_required: int,
) -> Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]]:
    combo_missing_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]] = (
        defaultdict(Counter)
    )
    missing_count = 0
    for key in combo_model_map:
        for model_pair in all_model_pairs:
            if combo_model_map[key][model_pair] < num_required:
                combo_missing_model_map[key][model_pair] += (
                    num_required - combo_model_map[key][model_pair]
                )
                missing_count += num_required - combo_model_map[key][model_pair]
    print("-" * 20 + f"Missing {missing_count} Model Pairs" + "-" * 20)
    print()
    return combo_missing_model_map


# temporally used for making sure unique (env, agents, models) setting; need to change
# according to the Counter in the case needing to run multiple experiments for one setting
def get_missing_model_combo_map(
    combo_missing_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name]]],
    all_combos_map: Dict[str, EnvAgentComboStorage],
) -> Dict[tuple[LLM_Name, LLM_Name], List[tuple[str, str, str]]]:
    missing_model_combo_map = defaultdict(list)
    for combo_pk in combo_missing_model_map:
        model_counter = combo_missing_model_map[combo_pk]
        for model_pair in model_counter:
            model_pair_key = (model_pair[1], model_pair[2])
            combo_model = all_combos_map[combo_pk]
            missing_model_combo_map[model_pair_key].append(
                (
                    combo_model.env_id,
                    combo_model.agent_ids[0],
                    combo_model.agent_ids[1],
                )
            )
    print("-" * 20 + "Missing Model to Combo Map" + "-" * 20)
    for key in missing_model_combo_map:
        print(f"Model pair: {key}")
        print(f"Number of missing combos: {len(missing_model_combo_map[key])}")
    return missing_model_combo_map


def yield_env_agent_combo(
    combo_ids: list[tuple[str, str, str]], model_names: dict[str, LLM_Name]
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    for combo_id in combo_ids:
        env_id, agent_id1, agent_id2 = combo_id
        env_profile = EnvironmentProfile.get(env_id)
        env = ParallelSotopiaEnv(
            env_profile=env_profile,
            model_name=model_names["env"],
            action_order="round-robin",
            evaluators=[
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            terminal_evaluators=[
                ReachGoalLLMEvaluator(
                    model_names["env"],
                    EvaluationForTwoAgents[SotopiaDimensions],
                ),
            ],
        )
        agent_profiles = [AgentProfile.get(id) for id in (agent_id1, agent_id2)]

        agents = [
            LLMAgent(agent_profile=agent_profile, model_name=agent_model)
            for agent_profile, agent_model in zip(
                agent_profiles,
                [model_names["agent1"], model_names["agent2"]],
            )
        ]
        yield env, agents


@gin.configurable
def re_run_missing_episodes(
    combo_with_models: dict[tuple[LLM_Name, LLM_Name], list[tuple[str, str, str]]],
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "agent1": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
    },
    batch_size: int = 5,
    verbose: bool = False,
) -> None:
    # pioritize gpt4 vs gpt3.5
    # pioritize gpt3.5 vs gpt3
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)
    combo_and_models_to_run = combo_with_models[
        (model_names["agent1"], model_names["agent2"])
    ]
    env_agent_combo_iter_length = len(combo_and_models_to_run)
    print(f"total missing episodes: {env_agent_combo_iter_length}")

    env_agent_combo_iter = yield_env_agent_combo(combo_and_models_to_run, model_names)
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []
    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        push_to_db=True,
                        tag="missing_episodes",
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        push_to_db=True,
                        tag="missing_episodes",
                    )
                )
            return


def main(_: Any) -> None:
    parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    all_episodes = get_all_episodes()
    all_combos_map = get_all_env_agent_combos(0, 5)
    combo_model_map = get_combo_model_map(all_episodes, all_combos_map)
    all_model_pairs = get_all_model_pairs(combo_model_map)
    combo_missing_model_map = get_all_missing_model_pairs(
        combo_model_map, all_model_pairs, 1
    )
    missing_model_combo_map = get_missing_model_combo_map(
        combo_missing_model_map, all_combos_map
    )
    re_run_missing_episodes(missing_model_combo_map)


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended "
        "to suffixes given via `--gin_file`. If a file appears in. Only the "
        "first prefix that produces a valid path for each suffix will be "
        "used.",
    )

    run(main)
