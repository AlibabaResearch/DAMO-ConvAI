from sotopia.database import EpisodeLog, AnnotationForEpisode
from typing import List, Tuple, Dict
from copy import deepcopy
from collections import defaultdict


def aggregate_reasoning(reasonings: List[str]) -> str:
    return "\n".join(reasonings)


def aggregate_rewards(
    rewards: List[Tuple[float, Dict[str, float]]],
) -> Tuple[float, Dict[str, float]]:
    def average_dict(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
        keys = dict_list[0].keys()
        return {k: sum(d[k] for d in dict_list) / len(dict_list) for k in keys}

    def average_list(item: List[float]) -> float:
        return sum(item) / len(item)

    ret_rewards = (
        average_list([r[0] for r in rewards]),
        average_dict([r[1] for r in rewards]),
    )
    return ret_rewards


def map_human_annotations_to_episode_logs(
    human_annotation: list[AnnotationForEpisode],
    return_model_episodes: bool = False,
    aggregate: bool = False,
) -> dict[str, EpisodeLog | tuple[EpisodeLog, EpisodeLog]]:
    """
    Map human annotations to corresponding episode logs.

    Args:
        human_annotations (List[AnnotationForEpisode]): List of annotations for episodes.
        return_model_episodes (bool, optional): If True, returns a tuple of human and model episodes. Defaults to False.
        aggregate (bool, optional): If True, aggregates reasoning and rewards across all human annotations per model episode. The primary key will be the model episode. Defaults to False.

    Returns:
        Dict[str, Union[EpisodeLog, Tuple[EpisodeLog, EpisodeLog]]]: A dictionary mapping episode primary keys to EpisodeLog objects or tuples of (human_episode, model_episode) depending on
        return_model_episodes and aggregate flags.
    """

    model_human_pk_mapping: Dict[str, List[str]] = defaultdict(list)
    for annotation in human_annotation:
        model_episode_pk = annotation.episode
        human_episode_pk = annotation.pk
        assert human_episode_pk is not None
        model_human_pk_mapping[model_episode_pk].append(human_episode_pk)

    ep_dict: Dict[str, EpisodeLog | Tuple[EpisodeLog, EpisodeLog]] = {}

    if aggregate:
        for model_episode_pk, human_episode_pks in model_human_pk_mapping.items():
            all_human_rewards = []
            all_human_reasonings = []

            for human_episode_pk in human_episode_pks:
                annotation = AnnotationForEpisode.get(pk=human_episode_pk)

                human_reasoning = annotation.reasoning
                human_rewards = annotation.rewards

                if any(isinstance(r, float) for r in human_rewards):
                    print("Rewards are not in the correct format")
                    print(human_rewards)
                    continue

                all_human_rewards.append(human_rewards)
                all_human_reasonings.append(human_reasoning)

            episode = EpisodeLog.get(pk=model_episode_pk)
            human_episode = deepcopy(episode)
            human_reasoning = aggregate_reasoning(all_human_reasonings)

            human_rewards = [
                aggregate_rewards([r[0] for r in all_human_rewards]),  # type: ignore
                aggregate_rewards([r[1] for r in all_human_rewards]),  # type: ignore
            ]
            if human_reasoning:
                human_episode.reasoning = human_reasoning
            if human_rewards:
                human_episode.rewards = human_rewards

            if return_model_episodes:
                ep_dict[model_episode_pk] = (human_episode, episode)
            else:
                ep_dict[model_episode_pk] = human_episode
        return ep_dict

    for annotation in human_annotation:
        episode_pk = annotation.episode
        episode = EpisodeLog.get(pk=episode_pk)
        human_episode_pk = annotation.pk
        assert human_episode_pk is not None

        human_reasoning = annotation.reasoning
        human_rewards = annotation.rewards

        human_episode = deepcopy(episode)
        human_episode.reasoning = human_reasoning
        human_episode.rewards = human_rewards

        if return_model_episodes:
            ep_dict[human_episode_pk] = (human_episode, episode)
        else:
            ep_dict[human_episode_pk] = human_episode

    return ep_dict
