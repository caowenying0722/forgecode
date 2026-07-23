'''Provider-neutral hook state objects.'''

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from forge.runtime.state import ToolCall, TurnResult
from forge.tools.base import ToolEffect, ToolResult


HookEvent = Literal[
    'user_prompt_submit',
    'pre_tool_use',
    'permission_request',
    'permission_denied',
    'post_tool_use',
    'stop',
]


@dataclass(frozen=True, slots=True)
class HookContext:
    event: HookEvent
    root: Path
    prompt: str | None = None
    tool_call: ToolCall | None = None
    effect: ToolEffect | None = None
    tool_result: ToolResult | None = None
    turn_result: TurnResult | None = None
    duration_seconds: float | None = None
    permission_mode: str | None = None
    metadata: dict[str, Any] | None = None

    def for_event(self, event: HookEvent, **updates: Any) -> HookContext:
        return replace(self, event=event, **updates)


@dataclass(frozen=True, slots=True)
class HookResult:
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    stop: bool = False
    metadata: dict[str, Any] | None = None

    @property
    def blocks_tool(self) -> bool:
        return self.tool_result is not None


@dataclass(frozen=True, slots=True)
class RegisteredHook:
    name: str
    events: tuple[HookEvent, ...]
    description: str = ''
    enabled: bool = True
