from redis_om import JsonModel
from redis_om.model.model import Field


class EnvAgentComboStorage(JsonModel):
    env_id: str = Field(default_factory=lambda: "", index=True)
    agent_ids: list[str] = Field(default_factory=lambda: [], index=True)
