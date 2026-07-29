'''Tests for centralized tool execution middleware.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from forge.runtime.state import ToolCall
from forge.runtime.tool_runner import ToolBatchState
from forge.runtime.tool_targets import mutation_target_paths
from forge.runtime.tool_executor import (
    PermissionMiddleware,
    ToolExecutionLogger,
    ToolExecutor,
)
from forge.hooks import TodoPlanningHook
from forge.hooks.registry import HookRegistry
from forge.hooks.state import HookContext
from forge.tools.base import Tool, ToolInput, ToolRegistry, ToolResult
from forge.tools.memory import create_memory_tools


class EmptyInput(ToolInput):
    pass


class ReadOnlyTool(Tool[EmptyInput]):
    name = 'read_sample'
    description = 'Read sample.'
    input_model = EmptyInput

    async def execute(self, arguments: EmptyInput) -> ToolResult:
        del arguments
        return ToolResult.ok('Read sample.', content='sample')


class WriteTool(Tool[EmptyInput]):
    name = 'write_sample'
    description = 'Write sample.'
    input_model = EmptyInput
    effect = 'workspace_write'

    async def execute(self, arguments: EmptyInput) -> ToolResult:
        del arguments
        (self.root / 'sample.txt').write_text('changed', encoding='utf-8')
        return ToolResult.ok('Wrote sample.')


class CommandInput(ToolInput):
    command: str


class CommandTool(Tool[CommandInput]):
    name = 'run_command'
    description = 'Run command.'
    input_model = CommandInput
    effect = 'process'

    async def execute(self, arguments: CommandInput) -> ToolResult:
        return ToolResult.ok(f'Ran {arguments.command}.')


class VerifyLikeTool(CommandTool):
    name = 'verify'


def run(coro):
    return asyncio.run(coro)


def test_tool_executor_allows_trusted_tools_and_logs_result(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry([ReadOnlyTool(tmp_path)])
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('trusted'),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_read', 'read_sample', {}))
    )

    assert record.result.success is True
    assert record.result.content == 'sample'
    log = json.loads(
        (tmp_path / '.forge' / 'logs' / 'tools.jsonl').read_text(
            encoding='utf-8'
        )
    )
    assert log['tool'] == 'read_sample'
    assert log['success'] is True
    assert log['permission_mode'] == 'trusted'


def test_strict_permission_blocks_workspace_write_before_execution(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry([WriteTool(tmp_path)])
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('strict'),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_write', 'write_sample', {}))
    )

    assert record.result.success is False
    assert record.result.error is not None
    assert record.result.error.code == 'permission_denied'
    assert not (tmp_path / 'sample.txt').exists()


def test_strict_permission_runs_workspace_write_when_approved(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry([WriteTool(tmp_path)])
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('strict', approver=lambda *_: True),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_write', 'write_sample', {}))
    )

    assert record.result.success is True
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'changed'


def test_strict_session_approval_reuses_tool_scope(tmp_path: Path) -> None:
    approvals: list[str] = []

    def approve(*_):
        approvals.append('asked')
        return 'allow_session'

    registry = ToolRegistry([WriteTool(tmp_path)])
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('strict', approver=approve),
        logger=ToolExecutionLogger(tmp_path),
    )

    first = run(
        executor.execute(ToolCall(0, 'toolu_write_1', 'write_sample', {}))
    )
    second = run(
        executor.execute(ToolCall(0, 'toolu_write_2', 'write_sample', {}))
    )

    assert first.result.success is True
    assert second.result.success is True
    assert approvals == ['asked']


def test_switching_permission_mode_clears_session_approvals(
    tmp_path: Path,
) -> None:
    approvals: list[str] = []

    def approve(*_):
        approvals.append('asked')
        return 'allow_session'

    permission = PermissionMiddleware('strict', approver=approve)
    executor = ToolExecutor(
        ToolRegistry([WriteTool(tmp_path)]),
        root=tmp_path,
        permission=permission,
        logger=ToolExecutionLogger(tmp_path),
    )
    call = ToolCall(0, 'toolu_write_1', 'write_sample', {})

    first = run(executor.execute(call))
    permission.set_mode('trusted')
    permission.set_mode('strict')
    second = run(
        executor.execute(ToolCall(0, 'toolu_write_2', 'write_sample', {}))
    )

    assert first.result.success is True
    assert second.result.success is True
    assert approvals == ['asked', 'asked']


def test_user_denial_is_recoverable_when_prompt_was_available(
    tmp_path: Path,
) -> None:
    executor = ToolExecutor(
        ToolRegistry([WriteTool(tmp_path)]),
        root=tmp_path,
        permission=PermissionMiddleware(
            'strict', approver=lambda *_: 'deny'
        ),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_deny', 'write_sample', {}))
    )

    assert record.result.success is False
    assert record.result.metadata['permission_terminal'] is False


def test_user_denial_does_not_count_as_failed_workspace_edit() -> None:
    call = ToolCall(0, 'toolu_deny', 'write_sample', {})
    denial = ToolResult.fail(
        'permission_denied',
        'User denied this tool call.',
        metadata={
            'permission_denied': True,
            'permission_terminal': False,
        },
    )
    batch = ToolBatchState(
        workspace_writes=[(0, call, denial, False)],
    )

    assert batch.pending_write_results(reverted_to_baseline=False) == []


def test_auto_mode_approves_workspace_write_without_prompt(
    tmp_path: Path,
) -> None:
    executor = ToolExecutor(
        ToolRegistry([WriteTool(tmp_path)]),
        root=tmp_path,
        permission=PermissionMiddleware('auto'),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_auto', 'write_sample', {}))
    )

    assert record.result.success is True


def test_auto_mode_prompts_for_risky_process_command(tmp_path: Path) -> None:
    asked: list[str] = []
    executor = ToolExecutor(
        ToolRegistry([CommandTool(tmp_path)]),
        root=tmp_path,
        permission=PermissionMiddleware(
            'auto',
            approver=lambda call, _effect: (
                asked.append(str(call.arguments['command'])) or 'allow_once'
            ),
        ),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(
            ToolCall(
                0,
                'toolu_risky',
                'run_command',
                {'command': 'git push origin main'},
            )
        )
    )

    assert record.result.success is True
    assert asked == ['git push origin main']


def test_auto_mode_checks_verify_command_for_risk(tmp_path: Path) -> None:
    asked: list[str] = []
    executor = ToolExecutor(
        ToolRegistry([VerifyLikeTool(tmp_path)]),
        root=tmp_path,
        permission=PermissionMiddleware(
            'auto',
            approver=lambda call, _effect: (
                asked.append(str(call.arguments['command'])) or 'deny'
            ),
        ),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(
            ToolCall(
                0,
                'toolu_verify_risky',
                'verify',
                {'command': 'git push origin main'},
            )
        )
    )

    assert record.result.success is False
    assert asked == ['git push origin main']


def test_todo_planning_hook_blocks_complex_write_before_todo(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry([WriteTool(tmp_path)])
    planning = TodoPlanningHook()
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('trusted'),
        logger=ToolExecutionLogger(tmp_path),
        hooks=HookRegistry([planning, ToolExecutionLogger(tmp_path)]),
    )
    run(
        executor.hooks.run(
            HookContext(
                event='user_prompt_submit',
                root=tmp_path,
                prompt='帮我完整实现权限 hook 系统',
                metadata={'todo_required': True},
            )
        )
    )

    record = run(
        executor.execute(ToolCall(0, 'toolu_write', 'write_sample', {}))
    )

    assert record.result.success is False
    assert record.result.error is not None
    assert record.result.error.code == 'todo_required'
    assert not (tmp_path / 'sample.txt').exists()


def test_memory_write_goes_through_permission_and_logging(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(create_memory_tools(tmp_path))
    executor = ToolExecutor(
        registry,
        root=tmp_path,
        permission=PermissionMiddleware('readonly'),
        logger=ToolExecutionLogger(tmp_path),
    )

    record = run(
        executor.execute(
            ToolCall(
                0,
                'toolu_memory',
                'memory_write',
                {'name': 'testing', 'content': 'Use pytest.'},
            )
        )
    )

    assert record.result.success is False
    assert record.result.error is not None
    assert record.result.error.code == 'permission_denied'
    assert not (tmp_path / '.forge' / 'memory' / 'testing.md').exists()
    log = json.loads(
        (tmp_path / '.forge' / 'logs' / 'tools.jsonl').read_text(
            encoding='utf-8'
        )
    )
    assert log['event'] == 'permission_denied'
    assert log['tool'] == 'memory_write'
    assert log['permission_mode'] == 'readonly'


def test_mutation_targets_are_shared_for_patch_tracking_and_recovery() -> None:
    call = ToolCall(
        0,
        'toolu_patch',
        'apply_patch',
        {
            'path': 'forge\\runtime\\agent_loop.py',
            'patch': (
                '*** Begin Patch\n'
                '*** Update File: forge/runtime/agent_loop.py\n'
                '*** Add File: forge/runtime/new_role.py\n'
                '*** End Patch'
            ),
        },
    )

    assert mutation_target_paths(call) == (
        'forge/runtime/agent_loop.py',
        'forge/runtime/new_role.py',
    )
