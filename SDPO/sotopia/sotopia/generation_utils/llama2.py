from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import together
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, root_validator


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: dict[str, str]) -> BaseMessage:
    text = _dict["text"]
    return AIMessage(content=text)


def _make_prompt_from_dict(dialog: List[dict[str, str]]) -> str:
    """
    Follow chat format in https://docs.together.ai/docs/python-chat
    example: together complete "<human>: List the best restaurants in SF\n<bot>: "
    python example: together.Complete.create(prompt=(above), model=model_name)
    The convertion is similar to llama2 official, EXCEPT not using special tags:
    https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
    """
    user_tag = "<human>"
    assistant_tag = "<bot>"
    tag_sep = ": "
    conv_sep = "\n"

    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": dialog[0]["content"] + conv_sep + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    ret = ""
    for d in dialog:
        tt = user_tag
        if d["role"] == "user":
            tt = user_tag
        elif d["role"] == "assistant":
            tt = assistant_tag
        ret += tt + tag_sep + d["content"].strip() + conv_sep
    ret += assistant_tag + tag_sep
    return ret


logger = logging.getLogger(__name__)


class Llama2(BaseChatModel):
    client: type[together.Complete] = together.Complete  #: :meta private:
    model_name: str = "togethercomputer/llama-2-7b-chat"
    """Model name to use."""
    # default Together params
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.0
    start: bool = False
    _llm_type: str = "llama2"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        together_api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        together.api_key = together_api_key
        values["client"] = together.Complete
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, params = self._create_message_dicts(messages, stop)
        response = self.client.create(prompt=prompt, **params)
        chat_result = self._create_chat_result(response)
        return chat_result

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Together API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
        }
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        message_prompt = _make_prompt_from_dict(message_dicts)
        return message_prompt, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["output"]["choices"]:
            message = _convert_dict_to_message(res)
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        sync_run_manager = cast(CallbackManagerForLLMRun, run_manager)
        return self._generate(messages, stop, sync_run_manager)
