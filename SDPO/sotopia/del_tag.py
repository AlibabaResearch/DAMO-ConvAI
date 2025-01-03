from sotopia.database.logs import EpisodeLog

model = "custom/sft@http://localhost:8000/v1/"
partner_model = "gpt-4o-mini"
task = "pos"
tag = f"benchmark_{model}_{partner_model}_gpt-4o_{task}_trial0"
tag = "benchmark_custom/eto@http://localhost:8000/v1/_gpt-4o_gpt-4o_all_trial1"
episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()

print(len(episodes))
for episode in episodes:
    pk = episode.pk
    EpisodeLog.delete(pk)