from pydantic import BaseModel, Field, validator


class RenderContext(BaseModel):
    viewer: str = Field(
        default="human",
        description="The viewer of the rendered string."
        "Must be one of ['human', 'environment', 'agent_i']"
        "where i is a non-negative integer, the index of the agent in the episode log.",
    )

    verbose: bool = Field(
        default=False,
        description="Whether to render the verbose version of the string.",
    )
    tags_to_render: list[str] = Field(
        default=[], description="The special tags to render."
    )

    @validator("viewer")
    def viewer_must_be_valid(cls, v: str) -> str:
        if v.startswith("agent_"):
            try:
                agent_idx = int(v.split("_")[1])
                assert agent_idx >= 0
            except Exception:
                raise ValueError(
                    "If viewer is an agent, it must be of the form agent_i, where i is a non-negative integer."
                )
        elif v not in ["human", "environment"]:
            raise ValueError(
                "Viewer must be one of ['human', 'environment', 'agent_i']"
                "where i is a non-negative integer, the index of the agent in the episode log."
            )
        return v


class BaseRenderer(object):
    def __call__(self, input_string: str, context: RenderContext) -> str:
        return input_string
