from redis_om import JsonModel
from redis_om.model.model import Field

DEFAULT_EXPIRE_TIME = 60 * 60 * 24 * 7  # 7 day


class AutoExpireMixin(JsonModel):
    expire_time: int = Field(default=DEFAULT_EXPIRE_TIME, index=True)

    def save(self) -> None:
        super().save()
        self.expire(self.expire_time)
