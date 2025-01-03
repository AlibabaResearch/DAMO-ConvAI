"""XML Renderer for background, goal, observation, etc."""

from io import StringIO
from typing import cast

from beartype.door import is_bearable
from lxml import etree

from .base import BaseRenderer, RenderContext


def _render_xml(xml_node: etree._Element | str, context: RenderContext) -> str:
    if isinstance(xml_node, str):
        return xml_node
    else:
        if xml_node.tag in ["root", "p"] or xml_node.tag in context.tags_to_render:
            if context.viewer.startswith("agent_"):
                # For each agent, we only render the messages viewable by that agent
                all_visible_children = xml_node.xpath(
                    f"./node()[@viewer='{context.viewer}'] | ./node()[not(@viewer)]"
                )
                assert is_bearable(all_visible_children, list[etree._Element | str])
                cast(list[etree._Element | str], all_visible_children)
                return "".join(
                    _render_xml(child, context)
                    for child in all_visible_children  # type: ignore[attr-defined]
                )
            elif context.viewer == "human":
                # For human, we render the raw xml
                return etree.tostring(xml_node, pretty_print=True).decode("utf-8")
            elif context.viewer == "environment":
                # For environment, we render all text
                all_text = xml_node.xpath("//text()")
                return "".join(cast(list[str], all_text))
        # Add return statement for the case where none of the conditions are met
        return ""


class XMLRenderer(BaseRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.parser = etree.XMLParser(recover=True, encoding="utf-8")

    def __call__(
        self, xml_string: str, context: RenderContext = RenderContext()
    ) -> str:
        if not xml_string:
            return ""
        try:
            root = etree.fromstring(xml_string)
        except etree.XMLSyntaxError:
            # try wrapping the xml_string with a pair of root tags
            try:
                root = etree.fromstring(f"<root>{xml_string}</root>")
            except etree.XMLSyntaxError:
                # try escaping the xml_string
                table = str.maketrans(
                    {
                        "&": "&amp;",
                    }
                )
                try:
                    root = etree.parse(
                        StringIO(f"<root>{xml_string.translate(table)}</root>"),
                        self.parser,
                    ).getroot()
                except etree.XMLSyntaxError as e:
                    raise etree.XMLSyntaxError(
                        f"Failed to parse xml_string: {xml_string}"
                    ) from e

        return _render_xml(root, context)
