'''Built-in hooks for permissions and tool execution logging.'''

from __future__ import annotations

import json
from pathlib import Path
import re
from time import time
from typing import Callable, Literal

from forge.hooks.state import HookContext, HookResult
from forge.runtime.intent import infer_task_contract
from forge.runtime.state import ToolCall
from forge.runtime.tool_targets import mutation_target_paths
from forge.tools.base import ToolEffect, ToolResult


PermissionMode = Literal['trusted', 'auto', 'strict', 'readonly']
ApprovalDecision = Literal['allow_once', 'allow_session', 'deny']
PermissionApprover = Callable[
    [ToolCall, ToolEffect | None],
    ApprovalDecision | bool,
]


class PermissionHook:
    name = 'permission'
    events = ('pre_tool_use',)
    description = 'Enforce full, auto, ask, or read-only tool permissions.'

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
        self.session_approvals: set[str] = set()

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
        if self.mode == 'auto' and auto_approval_allowed(tool_call, effect):
            return HookResult(metadata={'permission_auto_approved': True})
        if self.mode in {'strict', 'auto'} and effect in {
            'workspace_write',
            'process',
        }:
            return self._request_approval(tool_call, effect)
        return HookResult()

    def _request_approval(
        self,
        tool_call: ToolCall,
        effect: ToolEffect | None,
    ) -> HookResult:
        key = approval_scope_key(tool_call)
        if key in self.session_approvals:
            return HookResult(metadata={'permission_session_approved': True})
        if self.approver is None:
            return HookResult(
                tool_result=permission_denied_result(
                    tool_call,
                    self.mode,
                    effect,
                    'interactive approval is unavailable',
                    terminal=True,
                )
            )
        decision = normalize_approval_decision(
            self.approver(tool_call, effect)
        )
        if decision == 'allow_session':
            self.session_approvals.add(key)
        if decision in {'allow_once', 'allow_session'}:
            return HookResult(
                metadata={
                    'permission_approved': True,
                    'approval_decision': decision,
                }
            )
        return HookResult(
            tool_result=permission_denied_result(
                tool_call,
                self.mode,
                effect,
                'user denied this tool call',
                terminal=False,
            )
        )

    def set_mode(self, mode: PermissionMode) -> None:
        if mode != self.mode:
            self.session_approvals.clear()
        self.mode = mode


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


def normalize_permission_mode(mode: str) -> PermissionMode:
    normalized = mode.strip().casefold()
    aliases = {
        'read only': 'readonly',
        'read-only': 'readonly',
        'ask': 'strict',
        'ask for approval': 'strict',
        'approve': 'auto',
        'approve for me': 'auto',
        'full': 'trusted',
        'full access': 'trusted',
        'full-access': 'trusted',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {'trusted', 'auto', 'strict', 'readonly'}:
        raise ValueError(
            'Permission mode must be one of: readonly, strict, auto, trusted.'
        )
    return normalized  # type: ignore[return-value]


def render_permission_notice(mode: PermissionMode) -> str:
    if mode == 'trusted':
        return (
            'Permission: Full Access. All available tools may run without '
            'approval prompts; workspace boundaries and tool safety checks '
            'still apply.'
        )
    if mode == 'auto':
        return (
            'Permission: Approve for me. Workspace edits and low-risk local '
            'commands are approved automatically; risky or external actions '
            'still ask.'
        )
    if mode == 'readonly':
        return (
            'Permission: Read Only. Only read-only tools may run; write and '
            'process tools are blocked.'
        )
    return (
        'Permission: Ask for approval. Read-only tools may run directly; '
        'write and process tools ask for confirmation before execution.'
    )


def normalize_approval_decision(
    decision: ApprovalDecision | bool,
) -> ApprovalDecision:
    if decision is True:
        return 'allow_once'
    if decision is False:
        return 'deny'
    if decision not in {'allow_once', 'allow_session', 'deny'}:
        return 'deny'
    return decision


def approval_scope_key(tool_call: ToolCall) -> str:
    if tool_call.name == 'run_command':
        command = str(tool_call.arguments.get('command', '')).strip()
        return f'run_command:{command}'
    if tool_call.name.startswith('mcp_'):
        return tool_call.name
    targets = mutation_target_paths(tool_call)
    if targets:
        return f'{tool_call.name}:{"|".join(targets)}'
    return tool_call.name


RISKY_COMMAND_PATTERN = re.compile(
    r'(?i)(?:'
    r'\b(?:rm|rmdir|del|remove-item|format|shutdown|reboot)\b|'
    r'\bgit\s+(?:push|clean|reset|checkout|restore)\b|'
    r'\b(?:curl|wget|ssh|scp|ftp)\b|'
    r'\b(?:npm|pnpm|yarn|pip|uv)\s+(?:install|uninstall|publish)\b|'
    r'\b(?:docker|kubectl|terraform|ansible)\b|'
    r'\b(?:deploy|publish)\b'
    r')'
)


def auto_approval_allowed(
    tool_call: ToolCall,
    effect: ToolEffect | None,
) -> bool:
    if tool_call.name.startswith('mcp_'):
        return effect == 'read_only'
    if effect == 'workspace_write':
        return True
    if effect != 'process':
        return True
    if tool_call.name == 'task':
        return True
    if tool_call.name not in {'run_command', 'verify'}:
        return False
    command = str(tool_call.arguments.get('command', ''))
    return RISKY_COMMAND_PATTERN.search(command) is None


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
