from pydantic.v1 import validator
from redis_om import EmbeddedJsonModel, JsonModel
from redis_om.model.model import Field

from .auto_expires_mixin import AutoExpireMixin


class MessageTransaction(EmbeddedJsonModel):
    timestamp_str: str = Field(index=True)
    sender: str = Field(index=True)
    message: str

    def to_tuple(self) -> tuple[float, str, str]:
        return (
            float(self.timestamp_str),
            self.sender,
            self.message,
        )


class SessionTransaction(AutoExpireMixin, JsonModel):
    session_id: str = Field(index=True)
    client_id: str = Field(index=True)
    server_id: str = Field(index=True)
    client_action_lock: str = Field(default="no action")
    message_list: list[MessageTransaction] = Field(
        description="""List of messages in this session.
    Each message is a tuple of (timestamp, sender_id, message)
    The message list should be sorted by timestamp.
    """
    )

    @validator("message_list")
    def validate_message_list(
        cls, v: list[MessageTransaction]
    ) -> list[MessageTransaction]:
        def _is_sorted(x: list[MessageTransaction]) -> bool:
            return all(
                float(x[i].timestamp_str) <= float(x[i + 1].timestamp_str)
                for i in range(len(x) - 1)
            )

        assert _is_sorted(v), "Message list should be sorted by timestamp"
        return v
