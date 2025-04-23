from redis_om import JsonModel
from redis_om.model.model import Field

from .auto_expires_mixin import AutoExpireMixin


class MatchingInWaitingRoom(AutoExpireMixin, JsonModel):
    timestamp: float = Field()
    client_ids: list[str] = Field(default_factory=lambda: [])
    session_ids: list[str] = Field(default_factory=lambda: [])
    session_id_retrieved: list[str] = Field(default_factory=lambda: [])
