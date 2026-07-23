'''Tests for centralized tool execution middleware.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from forge.runtime.state import ToolCall
from forge.runtime.tool_executor import (
    PermissionMiddleware,
    ToolExecutionLogger,
    ToolExecutor,
)
from forge.tools.base import Tool, ToolInput, ToolRegistry, ToolResult


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
