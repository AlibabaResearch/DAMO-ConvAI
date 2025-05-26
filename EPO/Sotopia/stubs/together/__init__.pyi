from typing import Any, Dict, List, Optional

class Complete:
    @classmethod
    def create(
        self,
        prompt: str,
        model: Optional[str] = "",
        max_tokens: Optional[int] = 128,
        stop: Optional[List[str]] = [],
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        repetition_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
    ) -> Dict[str, Any]: ...

api_key: str = ...

class Models:
    @classmethod
    def start(cls, model_name: str) -> None: ...
