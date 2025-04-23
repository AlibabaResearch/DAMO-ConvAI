# XMLRenderers
[Source](../sotopia/renderers/xml_renderer.py)

The input of a renderer is an XML string and a `RenderContext`. Two tags are rendered by default:

- root: the root tag
- p: the paragraph tag

For both tags, a `viewer` attribute can be used to specify whether the text inside the tag will be rendered or not according to `RenderContext`.

Here is how the `viewer` determines the visibility of the text:

| viewer | visibility |
| --- | --- |
| "environment" | render all text |
| "human" | render the raw xml |
| "agent_i" | render the text that is viewable by agent_i |


> [!WARNING]
> The input to the xml render should be valid xml strings. Currently, we automatically fix the following two issues:
> - The `xml_string` is not a tree (e.g. `<a></a><b></b>`). We will automatically wrap the string with `<root></root>` if it is not wrapped.
> - The `xml_string` is not escaped properly. We only escape `&`. Quotations, `<` and `>` are not escaped by us.
