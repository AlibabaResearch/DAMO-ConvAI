class FLAGS(object):
    gin_search_paths: list[str]
    gin_file: list[str]
    gin_bindings: list[str]


def DEFINE_multi_string(name: str, default: list[str] | None, help: str) -> None:
    pass


def DEFINE_list(name: str, default: list[str] | None, help: str) -> None:
    pass
