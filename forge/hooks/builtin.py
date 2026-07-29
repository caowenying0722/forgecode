'''Built-in hooks for permissions and tool execution logging.'''

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Callable

from forge.hooks.state import HookContext, HookResult
from forge.permissions.policy import (
    ApprovalChoice,
    ApprovalResponse,
    PermissionManager,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    normalize_permission_mode,
    render_permission_notice,
)
from forge.permissions.risk import classify_tool_call
from forge.runtime.intent import infer_task_contract
from forge.runtime.state import ToolCall
from forge.tools.base import ToolEffect, ToolResult


ApprovalDecision = str | bool
PermissionApprover = Callable[
    [PermissionRequest],
    ApprovalResponse | ApprovalDecision,
]


class PermissionHook:
    name = 'permission'
    events = ('pre_tool_use',)
    description = 'Enforce plan, supervised, or auto tool permissions.'

    def __init__(
        self,
        mode: str = 'supervised',
        approver: PermissionApprover | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.mode = normalize_permission_mode(mode)
        self.approver = approver
        self.enabled = enabled
        self.session_rules: list[PermissionRule] = []

    async def handle(self, context: HookContext) -> HookResult:
        tool_call = require_tool_call(context)
        effect = context.effect
        request = classify_tool_call(tool_call, effect)
        manager = PermissionManager(
            context.root,
            mode=self.mode,
            approval_handler=(
                self._approval_handler if self.approver is not None else None
            ),
        )
        manager.session_rules = list(self.session_rules)
        decision = await manager.authorize(request)
        self.session_rules = list(manager.session_rules)
        if decision.action == 'allow':
            metadata = {
                'permission_request': request.capability,
                'permission_risk': request.risk,
                'permission_source': decision.source,
            }
            if decision.source == 'auto':
                metadata['permission_auto_approved'] = True
            if decision.source == 'session':
                metadata['permission_session_approved'] = True
            return HookResult(metadata=metadata)
        return HookResult(
            tool_result=permission_denied_result(
                tool_call,
                self.mode,
                effect,
                decision.reason,
                terminal=decision.source == 'approval_unavailable',
            )
        )

    async def _approval_handler(
        self,
        request: PermissionRequest,
    ) -> ApprovalResponse:
        if self.approver is None:
            raise RuntimeError('approval handler called without approver')
        raw = self.approver(request)
        if isinstance(raw, ApprovalResponse):
            return raw
        decision = normalize_approval_decision(raw)
        return ApprovalResponse(decision)

    def set_mode(self, mode: str) -> None:
        normalized = normalize_permission_mode(mode)
        if normalized != self.mode:
            self.session_rules.clear()
        self.mode = normalized


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

    def configure(self, *, required: bool) -> None:
        self.required = required
        self.planned = False

    async def handle(self, context: HookContext) -> HookResult:
        if context.event == 'user_prompt_submit':
            metadata = context.metadata or {}
            if 'todo_required' in metadata:
                self.required = bool(metadata['todo_required'])
            else:
                self.required = False
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
    contract = infer_task_contract(text, workspace_available=True)
    return contract.requires_change and contract.requires_plan


def normalize_approval_decision(
    decision: ApprovalDecision,
) -> ApprovalChoice:
    if decision is True:
        return 'allow_once'
    if decision is False:
        return 'deny'
    if decision not in {
        'allow_once',
        'allow_session',
        'allow_project',
        'deny',
    }:
        return 'deny'
    return decision  # type: ignore[return-value]


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
