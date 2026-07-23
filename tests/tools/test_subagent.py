'''Tests for bounded supervised Explore Subagent.'''

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any

from forge.hooks.builtin import PermissionHook
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)
from forge.tools.subagent import (
    ExploreSubagent,
    ExploreSubagentInput,
    SUBAGENT_EXCLUDED_TOOLS,
    create_subagent_registry,
)


class FakeSubagentClient:
    def __init__(self, *responses: list[ModelStreamEvent]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.calls.append(
            {
                'messages': messages,
                'tools': tools,
                'system': system,
            }
        )
        for event in self.responses.pop(0):
            yield event


def test_explore_subagent_uses_main_tools_except_task_controls_and_reports(
    tmp_path,
) -> None:
    (tmp_path / 'sample.txt').write_text('old value\n', encoding='utf-8')
    read = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'sample.txt'},
    )
    client = FakeSubagentClient(
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=0)),
            ModelToolCallCompleted(tool_call=read),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=2)),
        ],
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=0)),
            ModelTextDelta(
                text=(
                    'relevant_files: sample.txt\n'
                    'evidence: sample.txt contains old value\n'
                    'suggested_edit_points: sample.txt'
                )
            ),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=8)),
        ],
    )
    subagent = ExploreSubagent(tmp_path, client)

    result = asyncio.run(
        subagent.run(
            ExploreSubagentInput(task='Inspect sample', max_rounds=2)
        )
    )

    assert result.success is True
    assert 'sample.txt contains old value' in result.content
    assert result.metadata['subagent'] == 'explore'
    assert result.metadata['tool_calls'] == ['read_file']
    tool_names = {tool['name'] for tool in client.calls[0]['tools']}
    assert {
        'list_directory',
        'find_files',
        'read_file',
        'grep',
        'git_status',
        'write_file',
        'write_file_chunk',
        'replace_text',
        'apply_patch',
        'run_command',
        'verify',
        'git_diff',
        'memory_list',
        'memory_read',
        'memory_write',
        'memory_update',
        'memory_delete',
    }.issubset(tool_names)
    assert tool_names.isdisjoint(SUBAGENT_EXCLUDED_TOOLS)
    assert 'Explore Subagent' in client.calls[0]['system']


def test_explore_subagent_tool_calls_go_through_permission_hooks(
    tmp_path,
) -> None:
    write = ToolCall(
        index=0,
        id='toolu_write',
        name='write_file',
        arguments={'path': 'sample.txt', 'content': 'bad'},
    )
    client = FakeSubagentClient(
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=0)),
            ModelToolCallCompleted(tool_call=write),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=2)),
        ],
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=0)),
            ModelTextDelta(text='open_questions: write was denied'),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=8)),
        ],
    )
    subagent = ExploreSubagent(
        tmp_path,
        client,
        permission=PermissionHook('readonly'),
    )

    result = asyncio.run(
        subagent.run(
            ExploreSubagentInput(task='Try unsafe write', max_rounds=2)
        )
    )

    assert result.success is True
    assert not (tmp_path / 'sample.txt').exists()
    log = json.loads(
        (tmp_path / '.forge' / 'logs' / 'tools.jsonl').read_text(
            encoding='utf-8'
        ).splitlines()[0]
    )
    assert log['agent'] == 'explore_subagent'
    assert log['tool'] == 'write_file'
    assert log['error_code'] == 'permission_denied'


def test_explore_subagent_can_write_when_permission_allows(
    tmp_path,
) -> None:
    write = ToolCall(
        index=0,
        id='toolu_write',
        name='write_file',
        arguments={'path': 'sample.txt', 'content': 'written by subagent\n'},
    )
    client = FakeSubagentClient(
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=0)),
            ModelToolCallCompleted(tool_call=write),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=10, output_tokens=2)),
        ],
        [
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=0)),
            ModelTextDelta(
                text='changes_made: wrote sample.txt\nverification: not run'
            ),
            ModelUsageUpdate(usage=TokenUsage(input_tokens=20, output_tokens=8)),
        ],
    )
    subagent = ExploreSubagent(
        tmp_path,
        client,
        permission=PermissionHook('trusted'),
    )

    result = asyncio.run(
        subagent.run(
            ExploreSubagentInput(task='Write a file', max_rounds=2)
        )
    )

    assert result.success is True
    assert (tmp_path / 'sample.txt').read_text(
        encoding='utf-8'
    ) == 'written by subagent\n'
    log = json.loads(
        (tmp_path / '.forge' / 'logs' / 'tools.jsonl').read_text(
            encoding='utf-8'
        ).splitlines()[0]
    )
    assert log['event'] == 'post_tool_use'
    assert log['tool'] == 'write_file'
    assert log['success'] is True


def test_create_subagent_registry_excludes_recursive_control_tools(
    tmp_path,
) -> None:
    registry = create_subagent_registry(tmp_path)

    assert 'write_file' in registry.names
    assert 'run_command' in registry.names
    assert 'verify' in registry.names
    assert 'memory_write' in registry.names
    assert set(registry.names).isdisjoint(SUBAGENT_EXCLUDED_TOOLS)
