from typing import AbstractSet, Any, Dict, Mapping, Optional, Union

from pydantic.fields import Undefined, UndefinedType
from pydantic.typing import NoArgAnyCallable
from redis_om import JsonModel

class NotFoundError(Exception):
    """Raised when a query found no results."""

def Field(
    default: Any = Undefined,
    *,
    default_factory: Optional[NoArgAnyCallable] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Union[
        AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any
    ] = None,
    include: Union[
        AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any
    ] = None,
    const: Optional[str] = None,
    gt: Optional[str] = None,
    ge: Optional[str] = None,
    lt: Optional[str] = None,
    le: Optional[str] = None,
    multiple_of: Optional[str] = None,
    min_items: Optional[str] = None,
    max_items: Optional[str] = None,
    min_length: Optional[str] = None,
    max_length: Optional[str] = None,
    allow_mutation: bool = True,
    regex: Optional[str] = None,
    primary_key: bool = False,
    sortable: Union[bool, UndefinedType] = Undefined,
    index: Union[bool, UndefinedType] = Undefined,
    full_text_search: Union[bool, UndefinedType] = Undefined,
    schema_extra: Optional[Dict[str, Any]] = None,
) -> Any: ...

class FindQuery:
    def all(self) -> list[JsonModel]: ...
