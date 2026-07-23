'''Centralized tool execution boundary and cross-cutting middleware.'''

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter, time
from typing import Callable, Literal

from forge.runtime.state import ToolCall
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.base import ToolEffect, ToolRegistry, ToolResult


PermissionMode = Literal['trusted', 'strict', 'readonly']
AutoCommitMode = Literal['off', 'ask', 'always']
PermissionApprover = Callable[[ToolCall, ToolEffect | None], bool]


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    result: ToolResult
    effect: ToolEffect | None
    duration_seconds: float
    permission_mode: PermissionMode


@dataclass(frozen=True, slots=True)
class AutoCommitConfig:
    mode: AutoCommitMode = 'off'


class PermissionMiddleware:
    '''Apply session-level permission policy before tools run.'''

    def __init__(
        self,
        mode: PermissionMode = 'strict',
        approver: PermissionApprover | None = None,
    ) -> None:
        self.mode = mode
        self.approver = approver

    def check(self, tool_call: ToolCall, effect: ToolEffect | None) -> ToolResult | None:
        if self.mode == 'trusted':
            return None
        if self.mode == 'readonly' and effect != 'read_only':
            return permission_denied_result(
                tool_call,
                self.mode,
                effect,
                'readonly mode allows only read-only tools',
            )
        if self.mode == 'strict' and effect in {'workspace_write', 'process'}:
            if self.approver is not None and self.approver(tool_call, effect):
                return None
            return permission_denied_result(
                tool_call,
                self.mode,
                effect,
                'user did not approve this tool call',
                terminal=True,
            )
        return None


class ToolExecutionLogger:
    '''Append JSONL audit records for every tool execution attempt.'''

    def __init__(self, root: Path) -> None:
        self.path = root.resolve() / '.forge' / 'logs' / 'tools.jsonl'

    def record(
        self,
        tool_call: ToolCall,
        record: ToolExecutionRecord,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'timestamp': time(),
            'tool': tool_call.name,
            'arguments': tool_call.arguments,
            'effect': record.effect,
            'success': record.result.success,
            'summary': record.result.summary,
            'error_code': (
                record.result.error.code
                if record.result.error is not None
                else None
            ),
            'duration_seconds': round(record.duration_seconds, 6),
            'permission_mode': record.permission_mode,
        }
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


class ToolExecutor:
    '''Run all model-requested tools through one policy and logging boundary.'''

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        root: Path,
        workspace_tracker: WorkspaceTracker | None = None,
        permission: PermissionMiddleware | None = None,
        logger: ToolExecutionLogger | None = None,
        auto_commit: AutoCommitConfig | None = None,
    ) -> None:
        self.registry = registry
        self.root = root.resolve()
        self.workspace_tracker = workspace_tracker
        self.permission = permission or PermissionMiddleware()
        self.logger = logger or ToolExecutionLogger(self.root)
        self.auto_commit = auto_commit or AutoCommitConfig()

    def effect(self, name: str) -> ToolEffect | None:
        return self.registry.effect(name)

    async def execute(self, tool_call: ToolCall) -> ToolExecutionRecord:
        effect = self.effect(tool_call.name)
        if effect == 'workspace_write' and self.workspace_tracker is not None:
            self.workspace_tracker.watch_paths(mutation_target_paths(tool_call))

        started = perf_counter()
        denied = self.permission.check(tool_call, effect)
        if denied is None:
            result = await self.registry.execute(
                tool_call.name,
                tool_call.arguments,
            )
        else:
            result = denied
        record = ToolExecutionRecord(
            result=result,
            effect=effect,
            duration_seconds=perf_counter() - started,
            permission_mode=self.permission.mode,
        )
        self.logger.record(tool_call, record)
        return record


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


def mutation_target_paths(tool_call: ToolCall) -> tuple[str, ...]:
    raw_paths: list[object] = []
    arguments = tool_call.arguments
    for key in ('path', 'target_path'):
        if key in arguments:
            raw_paths.append(arguments[key])
    if tool_call.name == 'apply_patch':
        for key in ('paths', 'changed_paths'):
            value = arguments.get(key)
            if isinstance(value, list):
                raw_paths.extend(value)
    return tuple(str(path) for path in raw_paths if str(path).strip())
