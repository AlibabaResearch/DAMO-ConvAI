from evaluate_existing_episode import run_async_server_in_batch_aevaluate
from sotopia.database import (
    map_human_annotations_to_episode_logs,
    AnnotationForEpisode,
    EpisodeLog,
)
from typer import Typer
import numpy as np
import pandas as pd
import scipy.stats as stats

from sotopia.database.serialization import get_rewards_from_episode

app = Typer()

target_model_patterns: list[list[str]] = [
    ["gpt-4", "gpt-4", "gpt-3.5-turbo"],
    ["gpt-4", "gpt-3.5-turbo", "gpt-4"],
    ["gpt-4", "gpt-3.5-turbo", "togethercomputer/llama-2-70b-chat"],
    ["gpt-4", "togethercomputer/llama-2-70b-chat", "gpt-3.5-turbo"],
]


def get_human_annotations(
    target_model_patterns: list[list[str]],
) -> list[AnnotationForEpisode]:
    episodes_with_human_annotation: list[AnnotationForEpisode] = []
    for pk in AnnotationForEpisode.all_pks():
        episode_human = AnnotationForEpisode.get(pk)
        episode_model = EpisodeLog.get(episode_human.episode)
        if episode_model.models in target_model_patterns:
            episodes_with_human_annotation.append(episode_human)
    return episodes_with_human_annotation


def get_dimension_scores(
    agent_index: int, episodes: list[EpisodeLog], dimension: str, overall: bool
) -> list[float]:
    if overall:
        return [
            get_rewards_from_episode(relevant_episode)[agent_index][0]
            for relevant_episode in episodes
        ]
    else:
        return [
            get_rewards_from_episode(relevant_episode)[agent_index][1][dimension]
            for relevant_episode in episodes
        ]


def get_dimension_correlation(
    human_annotations: list[EpisodeLog],
    machine_annotations: list[EpisodeLog],
    dimension: str,
) -> dict[str, float]:
    # check the data is present
    episodes_with_valid_rewards = [
        int(not isinstance(relevant_episode.rewards[0], float))
        for relevant_episode in machine_annotations
    ]
    assert sum(episodes_with_valid_rewards) == len(human_annotations), "Data is missing"
    overall = dimension == "overall"
    dimension_scores_agent1_human = get_dimension_scores(
        0, human_annotations, dimension, overall
    )
    dimension_scores_agent2_human = get_dimension_scores(
        1, human_annotations, dimension, overall
    )
    dimension_scores_agent1_machine = get_dimension_scores(
        0, machine_annotations, dimension, overall
    )
    dimension_scores_agent2_machine = get_dimension_scores(
        1, machine_annotations, dimension, overall
    )
    x = dimension_scores_agent1_human + dimension_scores_agent2_human
    y = dimension_scores_agent1_machine + dimension_scores_agent2_machine
    res = stats.pearsonr(x, y)
    spearman_res = stats.spearmanr(x, y)
    mse = ((np.array(x) - np.array(y)) ** 2).mean()

    return {
        "pearson_correlation": res.statistic,
        "pearson_pvalue": res.pvalue,
        "spearman_correlation": spearman_res.correlation,
        "spearman_pvalue": spearman_res.pvalue,
        "mse": mse,
    }


@app.command()
def evaluate_evaluator(
    batch_size: int = 10,
    model: str = "gpt-4",
    tag: str = "reeval_gpt4",
    push_to_db: bool = False,
    verbose: bool = False,
) -> None:
    relevant_dimension = [
        "believability",
        "relationship",
        "knowledge",
        "secret",
        "social_rules",
        "financial_and_material_benefits",
        "goal",
    ]
    human_annotations = get_human_annotations(target_model_patterns)
    human_annotation_dict = map_human_annotations_to_episode_logs(
        human_annotations, return_model_episodes=False, aggregate=True
    )
    to_re_evaluate_list = list(human_annotation_dict.keys())
    aggregate_human_annotations: list[EpisodeLog] = list(human_annotation_dict.values())  # type: ignore
    # Call the function with the specified parameters

    re_evaluated_episodes: list[EpisodeLog] = EpisodeLog.find(
        EpisodeLog.tag == tag
    ).all()  # type: ignore
    if len(re_evaluated_episodes) < len(to_re_evaluate_list):
        run_async_server_in_batch_aevaluate(
            tag=tag,
            model=model,  # type: ignore
            batch_size=batch_size,
            push_to_db=push_to_db,
            verbose=verbose,
            reeval_list=to_re_evaluate_list,
        )
    else:
        valid_episodes = [
            not isinstance(relevant_episode.rewards[0], float)
            for relevant_episode in re_evaluated_episodes
        ]
        to_re_evaluate_list = [
            episode.pk  # type: ignore
            for valid, episode in zip(valid_episodes, re_evaluated_episodes)
            if not valid
        ]
        while to_re_evaluate_list:
            run_async_server_in_batch_aevaluate(
                tag=tag,
                model=model,  # type: ignore
                batch_size=batch_size,
                push_to_db=push_to_db,
                verbose=verbose,
                reeval_list=to_re_evaluate_list,
            )
            to_re_evaluate_list = []
            re_evaluated_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()  # type: ignore
            valid_episodes = [
                not isinstance(relevant_episode.rewards[0], float)
                for relevant_episode in re_evaluated_episodes
            ]
            for valid, episode in zip(valid_episodes, re_evaluated_episodes):
                if not valid:
                    pk = episode.pk
                    assert isinstance(pk, str)
                    EpisodeLog.delete(pk)
                    to_re_evaluate_list.append(pk)

    correlation_list = []
    ordered_re_eval_episodes = []

    for human_annotated_episode in aggregate_human_annotations:
        for re_eval_episode in re_evaluated_episodes:
            assert isinstance(re_eval_episode, EpisodeLog)
            if (
                human_annotated_episode.environment == re_eval_episode.environment
                and human_annotated_episode.agents == re_eval_episode.agents
                and human_annotated_episode.models[1:] == re_eval_episode.models[1:]  # type: ignore
            ):
                ordered_re_eval_episodes.append(re_eval_episode)
                break

    for dimension in relevant_dimension:
        correlation = get_dimension_correlation(
            aggregate_human_annotations, ordered_re_eval_episodes, dimension
        )
        correlation_list.append(correlation)
    print("Correlation between human and machine")
    print(pd.DataFrame(correlation_list, index=relevant_dimension))


if __name__ == "__main__":
    app()
