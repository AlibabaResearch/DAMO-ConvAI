## Generation functions

The core of generating agent action and environment observation lies in the `agenerate` function:

```python
@gin.configurable
@beartype
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: BaseOutputParser[OutputType],
    temperature: float = 0.7,
) -> OutputType:
    input_variables = re.findall(r"(?<!{){([^{}]+)}(?!})", template)
```
