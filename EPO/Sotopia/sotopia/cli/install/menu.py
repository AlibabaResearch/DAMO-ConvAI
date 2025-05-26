# Intially created by:
# gbPagano <guilhermebpagano@gmail.com>
# https://github.com/gbPagano/rich_menu

from typing import Literal
import click
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


class Menu:
    def __init__(
        self,
        *options: str,
        start_index: int = 0,
        rule_title: str = "MENU",
        panel_title: str = "",
        color: str = "bold green",
        align: Literal["left", "center", "right"] = "center",
        selection_char: str = ">",
    ):
        self.options = options
        self.index = start_index
        self.rule_tile = rule_title
        self.panel_title = panel_title
        self.color = color
        self.align = align
        self.selection_char = selection_char

    def _get_click(self) -> str | None:
        match click.getchar():
            case "\r":
                return "enter"
            case "\x1b[B" | "s" | "S" | "àP":
                return "down"
            case "\x1b[A" | "w" | "W" | "àH":
                return "up"
            case "\x1b[D" | "a" | "A" | "àK":
                return "left"
            case "\x1b[C" | "d" | "D" | "àM":
                return "right"
            case "\x1b":
                return "exit"
            case _:
                return None

    def _update_index(self, key: str) -> None:
        if key == "down":
            self.index += 1
        elif key == "up":
            self.index -= 1

        if self.index > len(self.options) - 1:
            self.index = 0
        elif self.index < 0:
            self.index = len(self.options) - 1

    @property
    def _group(self) -> Group:
        menu = Text(justify="left")

        selected = Text(self.selection_char + " ", self.color)
        not_selected = Text(" " * (len(self.selection_char) + 1))
        selections = [not_selected] * len(self.options)
        selections[self.index] = selected

        for idx, option in enumerate(self.options):
            menu.append(Text.assemble(selections[idx], option + "\n"))
        menu.rstrip()

        panel = Panel.fit(menu)
        panel.title = Text(self.panel_title, self.color)
        if self.rule_tile:
            group = Group(
                Rule(self.rule_tile, style=self.color),
                Align(panel, self.align),
            )
        else:
            group = Group(
                Align(panel, self.align),
            )

        return group

    def _clean_menu(self) -> None:
        rule = 1 if self.rule_tile else 0
        for _ in range(len(self.options) + rule + 2):
            print("\x1b[A\x1b[K", end="")

    def ask(
        self, screen: bool = True, esc: bool = True, return_index: bool = False
    ) -> str | int:
        with Live(self._group, auto_refresh=False, screen=screen) as live:
            live.update(self._group, refresh=True)
            while True:
                try:
                    key = self._get_click()
                    if key == "enter":
                        break
                    elif key == "exit" and esc:
                        exit()
                    elif key in ["up", "down"]:
                        self._update_index(key)
                        live.update(self._group, refresh=True)
                except (KeyboardInterrupt, EOFError):
                    exit()

        if not screen:
            self._clean_menu()

        return self.options[self.index] if not return_index else self.index
