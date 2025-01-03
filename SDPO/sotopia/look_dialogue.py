import os
import sys
import rich
from collections import Counter
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"

model = "custom/dpo@http://localhost:8000/v1/"
partner_model = "custom/dpo@http://localhost:8000/v1/"
task = "all"
Episodes = EpisodeLog.find(EpisodeLog.tag == f"benchmark_{model}_{partner_model}_gpt-4o_all_trial9").all()
Episodes = sorted(Episodes, key=lambda episode: (episode.environment, episode.agents))

print(len(Episodes))

hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments

for e in Episodes:
    if e.environment in hard_envs:
        agent_profiles, conversation = e.render_for_humans()
        for agent_profile in agent_profiles:
            rich.print(agent_profile)
        for message in conversation:
            rich.print(message)

print("\n" + "-"*10 + "\n")

for e in Episodes:
    if e.environment not in hard_envs:
        agent_profiles, conversation = e.render_for_humans()
        for agent_profile in agent_profiles:
            rich.print(agent_profile)
        for message in conversation:
            rich.print(message)

# episode_pks = EpisodeLog.all_pks()
# episode_pks = list(episode_pks)
# print(len(episode_pks))

# test_ep = EpisodeLog.get(episode_pks[0])
# agent_profiles, conversation = test_ep.render_for_humans()
# for agent_profile in agent_profiles:
#     rich.print(agent_profile)
# for message in conversation:
#     rich.print(message)

# get the epilogs that contain the specified models
# model1 = "gpt-4"
# model2 = "gpt-4"
# model_comp1 = ["gpt-4", model1, model2]
# model_comp2 = ["gpt-4", model2, model1]

# gpt4_gpt4_eps = []
# for epid in episode_pks:
#     try:
#         curr_ep = EpisodeLog.get(epid)
#     except Exception:
#         continue
#     if curr_ep.models == model_comp1 or curr_ep.models == model_comp2:
#         gpt4_gpt4_eps.append(curr_ep)
# len(gpt4_gpt4_eps)

# agent_profiles, conversation = gpt4_gpt4_eps[0].render_for_humans()
# for agent_profile in agent_profiles:
#     rich.print(agent_profile)
# for message in conversation:
#     rich.print(message)