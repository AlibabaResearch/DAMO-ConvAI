import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State


logger = logging.getLogger("agent_frame")


def interactive_loop(
    task: tasks.Task,
    thought_agent: agents.LMAgent,
    action_agent: agents.LMAgent,
    env_config: Dict[str, Any]
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    # Reset the env and set the prompt for reason-model
    thought_ob, thought_state = env.reset()

    logger.info(f"\n{Fore.YELLOW}{thought_ob}{Fore.RESET}")
    
    first_thought: str = thought_agent.__call_thought__(thought_state.history)
    logger.info(f"\n{Fore.GREEN}{first_thought}{Fore.RESET}\n")

    # Set the prompt for action_agent
    action_ob, action_state = env.reset_action(first_thought)
    logger.info(f"\n{Fore.YELLOW}{action_ob}{Fore.RESET}")

    cur_step = 1
    while not action_state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        # Agent act
        try:
            action_output: str = action_agent.__call_action__(action_state.history)
            logger.info(f"\n{Fore.GREEN}{action_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            action_state.success = False
            action_state.finished = True
            action_state.terminate_reason = "exceeding maximum input length"
            break
        # Environment step
        observation, thought_state, action_state = env.step(first_thought, action_output)
        thought_output: str = thought_agent.__call_thought__(thought_state.history)
        first_thought = thought_output
        action_state.history.append({
            "role": "assistant",
            "content": f"{first_thought}",
        })
        if not action_state.finished:
            # Color the observation in blue
            logger.info(
                f"\n{Fore.BLUE}{observation}{Fore.RESET}\n"
            )
            logger.info(f"\n{Fore.GREEN}{thought_output}{Fore.RESET}\n")

        if action_state.finished:
            break

    if action_state.reward is not None:
        logger.info(
            f"Task finished in {action_state.steps} steps. Success: {action_state.success}. Reward: {action_state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {action_state.steps} steps. Success: {action_state.success}"
        )

    return action_state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    
    with open(os.path.join(args.agent_path, f"{args.thought_agent_config}.json")) as f:
        thought_agent_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.action_agent_config}.json")) as f:
        action_agent_config: Dict[str, Any] = json.load(f)
    
    if args.thought_model_name is not None:
        thought_agent_config['config']['thought_model_name'] = args.thought_model_name
    if args.action_model_name is not None:
        action_agent_config['config']['action_model_name'] = args.action_model_name

    output_path = os.path.join("outputs", thought_agent_config['config']['thought_model_name'].replace('/', '_'), action_agent_config['config']['action_model_name'].replace('/', '_'), args.exp_config + args.exp_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        # from webshop.web_agent_site.envs import WebAgentTextEnv
        from envs.webshop.src.webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)

    # Initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    
    # Initialize the agents
    thought_agent: agents.LMAgent = getattr(agents, thought_agent_config["agent_class"])(
        thought_agent_config["config"]
    )
    action_agent: agents.LMAgent = getattr(agents, action_agent_config["agent_class"])(
        action_agent_config["config"]
    )
    
    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")


    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        return

    # Run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                task, thought_agent, action_agent, env_config
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
    
    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # Calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--thought_agent_config",
        type=str,
        default="fastchat",
        help="Config of thought model.",
    )
    parser.add_argument(
        "--action_agent_config",
        type=str,
        default="openai",
        help="Config of action model.",
    )
    parser.add_argument(
        "--thought_model_name",
        type=str,
        help="Thought model name. It will override the 'thought_model_name' in agent_config"
    )
    parser.add_argument(
        "--thought_model_temperature",
        type=float,
        default=0.7,
        help="Thought model temperature. It will override the 'thought_model_temperature' in agent_config"
    )
    parser.add_argument(
        "--thought_model_top_p",
        type=float,
        default=0.7,
        help="Thought model top_p. It will override the 'thought_model_top_p' in agent_config"
    )
    parser.add_argument(
        "--action_model_name",
        type=str,
        help="Action model name. It will override the 'action_model_name' in agent_config"
    )
    parser.add_argument(
        "--action_model_temperature",
        type=float,
        default=0.0,
        help="Action model temperature. It will override the 'action_model_temperature' in agent_config"
    )
    parser.add_argument(
        "--action_model_top_p",
        type=float,
        default=0.0,
        help="Action model top_p. It will override the 'action_model_top_p' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

