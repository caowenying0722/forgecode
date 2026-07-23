'''Centralized tool execution boundary and cross-cutting middleware.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from forge.hooks.builtin import (
    PermissionApprover,
    PermissionHook,
    PermissionMode,
    ToolLoggingHook,
    normalize_permission_mode,
    permission_denied_result,
    render_permission_notice,
)
from forge.hooks.registry import HookRegistry
from forge.hooks.state import HookContext
from forge.runtime.state import ToolCall
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.base import ToolEffect, ToolRegistry, ToolResult


AutoCommitMode = Literal['off', 'ask', 'always']


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    result: ToolResult
    effect: ToolEffect | None
    duration_seconds: float
    permission_mode: PermissionMode


@dataclass(frozen=True, slots=True)
class AutoCommitConfig:
    mode: AutoCommitMode = 'off'


PermissionMiddleware = PermissionHook
ToolExecutionLogger = ToolLoggingHook


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
        hooks: HookRegistry | None = None,
        auto_commit: AutoCommitConfig | None = None,
    ) -> None:
        self.registry = registry
        self.root = root.resolve()
        self.workspace_tracker = workspace_tracker
        self.permission = permission or PermissionMiddleware()
        self.logger = logger or ToolExecutionLogger(self.root)
        self.hooks = hooks or HookRegistry([self.permission, self.logger])
        self.auto_commit = auto_commit or AutoCommitConfig()

    def effect(self, name: str) -> ToolEffect | None:
        return self.registry.effect(name)

    async def execute(self, tool_call: ToolCall) -> ToolExecutionRecord:
        effect = self.effect(tool_call.name)
        if effect == 'workspace_write' and self.workspace_tracker is not None:
            self.workspace_tracker.watch_paths(mutation_target_paths(tool_call))

        started = perf_counter()
        pre = await self.hooks.run(
            HookContext(
                event='pre_tool_use',
                root=self.root,
                tool_call=tool_call,
                effect=effect,
                permission_mode=self.permission.mode,
            )
        )
        active_call = pre.tool_call or tool_call
        result = pre.tool_result
        if result is None:
            result = await self.registry.execute(
                active_call.name,
                active_call.arguments,
            )
        duration = perf_counter() - started
        record = ToolExecutionRecord(
            result=result,
            effect=effect,
            duration_seconds=duration,
            permission_mode=self.permission.mode,
        )
        await self.hooks.run(
            HookContext(
                event=(
                    'permission_denied'
                    if result.metadata.get('permission_denied')
                    else 'post_tool_use'
                ),
                root=self.root,
                tool_call=active_call,
                effect=effect,
                tool_result=result,
                duration_seconds=duration,
                permission_mode=self.permission.mode,
            )
        )
        return record


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
