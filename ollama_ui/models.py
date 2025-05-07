from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from rich.console import Console
from rich.text import Text

ROLES = Literal["user", "assistant", "system", "tool"]
REASONING_TAGS = Literal["think", "thinking", "reasoning"]


@dataclass
class Message:
    """Represent ollama chat message and UI strucutred format."""

    role: ROLES
    content: str = ""
    main_text: str | None = None
    reasoning_text: str | None = None
    reasoning_tag: REASONING_TAGS | None = None
    in_reasoning: bool = False

    def rich_print(self):
        """Print the Message using Rich Text formatting."""
        # Define color scheme for different roles
        role_colors = {
            "user": "bright_blue",
            "assistant": "green",
            "system": "yellow",
            "tool": "cyan",
        }

        text = Text()

        # set role style
        color = role_colors.get(self.role, "white")

        # Role Header
        text.append(f"\n[{self.role.upper()}]\n", style=f"{color} bold")

        # Reasoning Text
        if self.reasoning_text:
            text.append(
                f"<{self.reasoning_tag}>\n{self.reasoning_text.strip()}\n",
                style=f"{color} italic dim",
            )

        # Main Text, System or Tool
        if self.main_text:
            text.append(f"{self.main_text.strip()}", style=color)
        else:
            text.append(f"{self.content.strip()}", style=color)

        # Create a console and print the text directly
        console = Console()
        console.print(text)

    def update_from_stream(self, stream: Iterable):
        """Incrementally update the Message content from a stream."""
        for chunk in stream:
            content = chunk["message"]["content"]
            self.content += content

            if not self.in_reasoning:
                for tag in get_args(REASONING_TAGS):
                    start_tag = f"<{tag}>"
                    if start_tag in self.content:  # search the whole current context
                        self.reasoning_tag = tag
                        self.in_reasoning = True
                        _, _, content = self.content.partition(start_tag)
                        self.main_text = ""  # reset
                        break

            if self.in_reasoning:
                start_tag = f"<{self.reasoning_tag}>"  # ensure start_tag exists
                end_tag = f"</{self.reasoning_tag}>"
                if end_tag in self.content:
                    reasoning, _, remaining = self.content.partition(end_tag)

                    self.reasoning_text = reasoning.replace(start_tag, "")
                    self.main_text = (self.main_text or "") + remaining
                    self.in_reasoning = False
                else:
                    self.reasoning_text = (self.reasoning_text or "") + content
            else:
                self.main_text = (self.main_text or "") + content

            # return the current instance after each chunk is processed
            yield self

        @classmethod
        def from_text(cls, role: ROLES, content: str):
            """Create a Message instance from raw text content."""
            instance = cls(role=role, content=content)

            # Check each possible tag in order
            for tag in get_args(REASONING_TAGS):
                start_tag = f"<{tag}>"
                end_tag = f"</{tag}>"

                if start_tag in content:
                    # Split into pre-reasoning and reasoning sections
                    before_reasoning, _, during_reasoning = content.partition(start_tag)
                    instance.main_text = before_reasoning.strip()
                    instance.reasoning_tag = tag

                    # Handle potential end tag
                    if end_tag in during_reasoning:
                        reasoning_content, _, after_reasoning = (
                            during_reasoning.partition(end_tag)
                        )
                        instance.reasoning_text = reasoning_content.strip()
                        instance.main_text += (
                            f" {after_reasoning.strip()}" if after_reasoning else ""
                        )
                    break
            else:
                # No tags found
                instance.main_text = content.strip()

            return instance
