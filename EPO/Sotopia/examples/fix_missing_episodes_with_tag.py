"""
Updated version of fix_missing_episodes.py
1. Add support to specify tags to fix
2. Now can automatically use correct models and tag when fixing

One sample usage can be:
python examples/fix_missing_episodes_with_tag.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \

"""

import asyncio
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, List, Set, Tuple, cast

import gin
from absl import flags
from absl.flags import FLAGS
from rich.logging import RichHandler
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

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
from sotopia.server import arun_one_script, run_async_server
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
def get_all_episodes(tags: List[str] = []) -> List[Tuple[EpisodeLog, str]]:
    # episode_pks: List[str] = list(EpisodeLog.all_pks())
    assert len(tags) != 0, "No episodes found"
    if len(tags) != 0:
        print("Using tag list: ", tags)
        episode_pks: List[Tuple[(str, str)]] = []
        for tag in tags:
            ep_list = list(EpisodeLog.find(EpisodeLog.tag == tag).all())
            episode_pks += [(ep.pk, ep.tag) for ep in ep_list]  # type: ignore

    all_episodes = []
    for pk, tag in tqdm(episode_pks):
        try:
            curr_ep = EpisodeLog.get(pk)
        except Exception as _:
            continue
        all_episodes.append((curr_ep, tag))
    print(f"all episodes loaded {len(all_episodes)}")
    return all_episodes


# all env-agent combos
def get_all_env_agent_combos(
    all_tags: List[str], start_combo_idx: int, end_combo_idx: int
) -> Dict[str, EnvAgentComboStorage]:
    experiment_env_pks = list(EnvironmentProfile.all_pks())
    # experiment_env_pks: List[str] = []
    assert len(all_tags) != 0, "No envs found"
    # for tag in all_tags:
    #     experiment_env_pks += [ep.pk for ep in EpisodeLog.find(EpisodeLog.tag == tag).all()]

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
    all_episodes: List[Tuple[EpisodeLog, str]],
    all_combos_map: Dict[str, EnvAgentComboStorage],
) -> Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]]:
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]] = (
        defaultdict(Counter)
    )

    bad_combos = []
    valid_count = 0
    invalid_count = 0
    bad_rewards_count = 0
    bad_gpt4_rewards_count = 0
    bad_combo_count = 0
    episodes_to_delete = []

    # iterate through episodes
    for i in tqdm(range(len(all_episodes))):
        curr_ep, curr_tag = all_episodes[i]
        bad_rewards = False

        # check if episode is valid
        if not curr_ep.models:
            invalid_count += 1
            episodes_to_delete.append(curr_ep.pk)
            continue

        # check if rewards are valid
        for idx, model in enumerate(curr_ep.models[1:]):
            if not isinstance(curr_ep.rewards[idx], tuple):
                bad_rewards = True
                break
        if bad_rewards:
            bad_rewards_count += 1
            episodes_to_delete.append(curr_ep.pk)
            if tuple(curr_ep.models) == ("gpt4", "gpt4", "gpt4"):
                bad_gpt4_rewards_count += 1
            continue

        # check if the length is too short
        rendered_ep = curr_ep.render_for_humans()[1]
        interaction_list = rendered_ep[1:-2]

        # check if the lite mode is erroneously used
        # if "background" in rendered_ep[0] and "lite" in curr_ep.tag:
        #     episodes_to_delete.append(curr_ep.pk)
        #     # print(f"{curr_ep.pk} has background!")
        #     bad_lite_mode_count += 1
        #     continue

        # check if the background is present
        # if not "lite" in curr_ep.tag:
        #     concat_text = "\n".join(rendered_ep[:-2])
        #     if "'s background" not in concat_text:
        #         print("Background not found in: ", curr_ep.pk)
        #         episodes_to_delete.append(curr_ep.pk)
        #         bad_background_count += 1
        #         continue

        if len(interaction_list) <= 5:
            # bad_rewards_count += 1
            episodes_to_delete.append(curr_ep.pk)
            continue

        # find combo pk by env pk and agent ids
        curr_combo_pk = find_combo_pk(
            curr_ep.environment,
            curr_ep.agents[0],
            curr_ep.agents[1],
            all_combos_map,
        )
        import copy

        if curr_combo_pk:
            combined = copy.deepcopy(curr_ep.models)
            combined.append(curr_tag)

            model_pair: tuple[LLM_Name, LLM_Name, LLM_Name, str] = cast(
                tuple[LLM_Name, LLM_Name, LLM_Name, str], tuple(combined)
            )
            combo_model_map[curr_combo_pk][model_pair] += 1
            valid_count += 1
        else:
            bad_combos.append(
                (curr_ep.environment, curr_ep.agents[0], curr_ep.agents[1])
            )
            # episodes_to_delete.append(curr_ep.pk)

            bad_combo_count += 1
    # print("Bad lite mode count: ", bad_lite_mode_count)
    # print("Bad background count: ", bad_background_count)
    # exit(0)
    print("-" * 20 + "Deleting Bad Combos" + "-" * 20)
    for ep_pk in episodes_to_delete:
        # TODO Do we actually need to delete these bad episodes? (I think yes, as this can help us to get the correct number of episodes)
        print("Delete episode: ", ep_pk)
        EpisodeLog.delete(ep_pk)

    print("-" * 20 + "Episode Parsing Summary" + "-" * 20)
    print(f"valid episodes: {valid_count}")
    print(f"invalid episodes (missing episode.models): {invalid_count}")
    print(f"bad combo: {bad_combo_count}")
    print(f"bad rewards: {bad_rewards_count}")
    print(f"bad gpt4 rewards: {bad_gpt4_rewards_count}")
    return combo_model_map


def get_all_model_pairs(
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]],
) -> Set[tuple[LLM_Name, LLM_Name, LLM_Name, str]]:
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
    combo_model_map: Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]],
    all_model_pairs: Set[tuple[LLM_Name, LLM_Name, LLM_Name, str]],
    num_required: int,
    all_combos_map: Dict[str, EnvAgentComboStorage] = {},
    add_missing_env: bool = False,
) -> Dict[str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]]:
    """
    all_combos_map: if add_missing_env is True, then we need to provide all combos map
    add_missing_env: if True, add missing env to the map, else just match the model pairs among selected tags
    """
    combo_missing_model_map: Dict[
        str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]
    ] = defaultdict(Counter)

    if add_missing_env:
        for combo_key in all_combos_map:
            if combo_key not in combo_model_map:
                combo_missing_model_map[combo_key] = Counter()
                for model_pair in all_model_pairs:
                    combo_missing_model_map[combo_key][model_pair] = 0
                print("Missing combo: ", combo_key)

    missing_count = 0
    for key in combo_model_map:
        for model_pair in all_model_pairs:
            model_tag_pair = model_pair
            # print(model_tag_pair, key, combo_model_map[key][model_tag_pair])
            if combo_model_map[key][model_tag_pair] < num_required:
                combo_missing_model_map[key][model_tag_pair] += (
                    num_required - combo_model_map[key][model_tag_pair]
                )
                missing_count += num_required - combo_model_map[key][model_tag_pair]
    print("-" * 20 + f"Missing {missing_count} Model Pairs" + "-" * 20)
    print()
    return combo_missing_model_map


# temporally used for making sure unique (env, agents, models) setting; need to change
# according to the Counter in the case needing to run multiple experiments for one setting
def get_missing_model_combo_map(
    combo_missing_model_map: Dict[
        str, Counter[tuple[LLM_Name, LLM_Name, LLM_Name, str]]
    ],
    all_combos_map: Dict[str, EnvAgentComboStorage],
) -> Dict[tuple[LLM_Name, LLM_Name, LLM_Name, str], List[tuple[str, str, str]]]:
    missing_model_combo_map = defaultdict(list)
    for combo_pk in combo_missing_model_map:
        model_counter = combo_missing_model_map[combo_pk]
        for model_pair in model_counter:
            model_pair_key = (
                model_pair[0],
                model_pair[1],
                model_pair[2],
                model_pair[3],
            )
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
    env_agent_ids: List[Tuple[str, str, str]] = [],
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "agent1": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
    },
    batch_size: int = 5,
    rerun_tag: str = "missing_episodes",
    verbose: bool = False,
) -> None:
    """
    1. Running script in full mode:
        arun_one_script(
            env=env_agent_combo[0],
            agent_list=env_agent_combo[1],
            model_dict=model_names,
            tag=tag,
            push_to_db=True,
        )
    2. Running normally:
        run_async_server(
            model_dict=model_names,
            sampler=BaseSampler[Observation, AgentAction](),
            env_agent_combo_list=env_agent_combo_batch,
            push_to_db=True,
            tag=rerun_tag,
        )
    """
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)

    combo_and_models_to_run = env_agent_ids

    env_agent_combo_iter_length = len(combo_and_models_to_run)
    print(f"Rerunning with model: {model_names} and tag: {rerun_tag}")
    print(f"total missing episodes: {env_agent_combo_iter_length}")

    env_agent_combo_iter = yield_env_agent_combo(combo_and_models_to_run, model_names)
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []

    def run_script() -> None:
        episode_futures = [
            arun_one_script(
                env=env_agent_combo[0],
                agent_list=env_agent_combo[1],
                model_dict=model_names,
                tag=rerun_tag,
                push_to_db=True,
            )
            for env_agent_combo in env_agent_combo_batch
        ]
        _ = asyncio.run(tqdm_asyncio.gather(*episode_futures, desc="Running one batch"))

    def run_interaction() -> None:
        asyncio.run(
            run_async_server(
                model_dict=model_names,
                sampler=BaseSampler[Observation, AgentAction](),
                env_agent_combo_list=env_agent_combo_batch,
                push_to_db=True,
                tag=rerun_tag,
                omniscient=omniscient_mode,
            )
        )

    script_mode = "script" in rerun_tag
    omniscient_mode = "omni" in rerun_tag

    run_func = run_script if script_mode else run_interaction
    print("Current mode: ", "script" if script_mode else "interaction")
    print("Current omniscient mode: ", omniscient_mode)

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
                run_func()
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                run_func()
            return


@gin.configurable
def rerun_missing_episodes(tags: List[str] = []) -> None:
    print("All tags to fix: ", tags)
    all_episodes = get_all_episodes(tags=tags)
    all_combos_map = get_all_env_agent_combos(tags, 0, 5)
    combo_model_map = get_combo_model_map(all_episodes, all_combos_map)

    all_model_pairs = get_all_model_pairs(combo_model_map)
    combo_missing_model_map = get_all_missing_model_pairs(
        combo_model_map,
        all_model_pairs,
        1,
        all_combos_map=all_combos_map,
        add_missing_env=True,
    )

    missing_model_combo_map = get_missing_model_combo_map(
        combo_missing_model_map, all_combos_map
    )

    for model_tag, episode_env_id in missing_model_combo_map.items():
        rerun_model_names = {
            "env": model_tag[0],
            "agent1": model_tag[1],
            "agent2": model_tag[2],
        }
        tag = model_tag[3]
        re_run_missing_episodes(
            env_agent_ids=episode_env_id,
            model_names=rerun_model_names,
            rerun_tag=tag,
        )


def main(_: Any) -> None:
    parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    rerun_missing_episodes()


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
