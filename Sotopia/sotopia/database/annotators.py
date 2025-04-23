from pydantic import Field
from redis_om import JsonModel


class Annotator(JsonModel):
    name: str = Field(index=True, required=True)
    email: str = Field(index=True, required=True)
