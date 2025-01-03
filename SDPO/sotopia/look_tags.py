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

tags = set()
pks = list(EpisodeLog.all_pks())
print(len(pks))
for i in range(len(pks)):
    try:
        e = EpisodeLog.get(pks[i])
        tags.add(e.tag)
    except:
        continue


print(tags)
# for tag in list(tags):
#     if tag != None and ("benchmark" not in tag or "eto" in tag):
#         episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
#         print(tag)
#         for episode in episodes:
#             pk = episode.pk
#             EpisodeLog.delete(pk)