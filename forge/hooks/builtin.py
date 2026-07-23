'''Built-in hooks for permissions and tool execution logging.'''

from __future__ import annotations

import json
from pathlib import Path
import re
from time import time
from typing import Callable, Literal

from forge.hooks.state import HookContext, HookResult
from forge.runtime.state import ToolCall
from forge.tools.base import ToolEffect, ToolResult


PermissionMode = Literal['trusted', 'strict', 'readonly']
PermissionApprover = Callable[[ToolCall, ToolEffect | None], bool]


class PermissionHook:
    name = 'permission'
    events = ('pre_tool_use',)
    description = 'Enforce trusted, strict, or readonly tool permission policy.'

    def __init__(
        self,
        mode: PermissionMode = 'strict',
        approver: PermissionApprover | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.mode = mode
        self.approver = approver
        self.enabled = enabled

    async def handle(self, context: HookContext) -> HookResult:
        tool_call = require_tool_call(context)
        effect = context.effect
        if self.mode == 'trusted':
            return HookResult()
        if self.mode == 'readonly' and effect != 'read_only':
            return HookResult(
                tool_result=permission_denied_result(
                    tool_call,
                    self.mode,
                    effect,
                    'readonly mode allows only read-only tools',
                )
            )
        if self.mode == 'strict' and effect in {'workspace_write', 'process'}:
            if self.approver is not None and self.approver(tool_call, effect):
                return HookResult(metadata={'permission_approved': True})
            return HookResult(
                tool_result=permission_denied_result(
                    tool_call,
                    self.mode,
                    effect,
                    'user did not approve this tool call',
                    terminal=True,
                )
            )
        return HookResult()


class ToolLoggingHook:
    name = 'tool_logging'
    events = ('post_tool_use', 'permission_denied')
    description = 'Append tool execution audit records to .forge/logs/tools.jsonl.'

    def __init__(
        self,
        root: Path,
        *,
        agent: str = 'main',
        enabled: bool = True,
    ) -> None:
        self.path = root.resolve() / '.forge' / 'logs' / 'tools.jsonl'
        self.agent = agent
        self.enabled = enabled

    async def handle(self, context: HookContext) -> HookResult:
        if context.tool_call is None or context.tool_result is None:
            return HookResult()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        result = context.tool_result
        payload = {
            'timestamp': time(),
            'event': context.event,
            'agent': self.agent,
            'tool': context.tool_call.name,
            'arguments': context.tool_call.arguments,
            'effect': context.effect,
            'success': result.success,
            'summary': result.summary,
            'error_code': (
                result.error.code if result.error is not None else None
            ),
            'duration_seconds': (
                None
                if context.duration_seconds is None
                else round(context.duration_seconds, 6)
            ),
            'permission_mode': context.permission_mode,
        }
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        return HookResult()


class TodoPlanningHook:
    name = 'todo_planning'
    events = ('user_prompt_submit', 'pre_tool_use', 'post_tool_use')
    description = (
        'Require todo_write before write or process tools on complex tasks.'
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.required = False
        self.planned = False

    async def handle(self, context: HookContext) -> HookResult:
        if context.event == 'user_prompt_submit':
            self.required = should_require_todo_plan(context.prompt or '')
            self.planned = False
            return HookResult(
                metadata={'todo_required': self.required}
            )
        if context.event == 'post_tool_use':
            if (
                context.tool_call is not None
                and context.tool_call.name == 'todo_write'
                and context.tool_result is not None
                and context.tool_result.success
            ):
                self.planned = True
            return HookResult()
        if context.event == 'pre_tool_use':
            tool_call = require_tool_call(context)
            if tool_call.name == 'todo_write':
                return HookResult()
            if (
                self.required
                and not self.planned
                and context.effect in {'workspace_write', 'process'}
            ):
                return HookResult(
                    tool_result=ToolResult.fail(
                        'todo_required',
                        (
                            'This task looks complex. Do not continue with a '
                            'prose-only plan or another write/process tool. '
                            'Call the todo_write tool next with a short '
                            'working plan before using write or process tools.'
                        ),
                        metadata={
                            'todo_required': True,
                            'terminal': False,
                        },
                    )
                )
        return HookResult()


def require_tool_call(context: HookContext) -> ToolCall:
    if context.tool_call is None:
        raise ValueError(f'{context.event} hook requires a tool_call.')
    return context.tool_call


def should_require_todo_plan(prompt: str) -> bool:
    text = prompt.strip()
    if not text:
        return False
    lowered = text.casefold()
    if re.search(r'\b(?:p0|p1|p2|priority|priorities|roadmap)\b', lowered):
        return True
    if re.search(
        r'\b(?:implement|refactor|architecture|migrate|integration)\b',
        lowered,
    ):
        return True
    if re.search(
        r'(?:实现|重构|迁移|架构|完整|逐一|优先级|规划|计划|系统|多代理|权限|hook|mcp)',
        text,
    ):
        return True
    return False


def normalize_permission_mode(mode: str) -> PermissionMode:
    normalized = mode.strip().casefold()
    if normalized not in {'trusted', 'strict', 'readonly'}:
        raise ValueError(
            'Permission mode must be one of: trusted, strict, readonly.'
        )
    return normalized  # type: ignore[return-value]


def render_permission_notice(mode: PermissionMode) -> str:
    if mode == 'trusted':
        return (
            'Permission: trusted. Read, write, and process tools may run '
            'without runtime permission blocking.'
        )
    if mode == 'readonly':
        return (
            'Permission: readonly. Only read-only tools may run; write and '
            'process tools are blocked.'
        )
    return (
        'Permission: strict. Read-only tools may run directly; write and '
        'process tools ask for confirmation before execution.'
    )


def permission_denied_result(
    tool_call: ToolCall,
    mode: PermissionMode,
    effect: ToolEffect | None,
    reason: str,
    *,
    terminal: bool = False,
) -> ToolResult:
    return ToolResult.fail(
        'permission_denied',
        f'Permission denied for {tool_call.name}: {reason}.',
        details={
            'tool': tool_call.name,
            'effect': effect,
            'permission_mode': mode,
            'terminal': terminal,
        },
        metadata={
            'permission_denied': True,
            'permission_terminal': terminal,
        },
    )
