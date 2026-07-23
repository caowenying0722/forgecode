'''Tests for bounded read-only Explore Subagent.'''

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)
from forge.tools.subagent import ExploreSubagent, ExploreSubagentInput


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


def test_explore_subagent_uses_only_read_only_tools_and_reports(
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
    assert tool_names == {
        'list_directory',
        'find_files',
        'read_file',
        'grep',
        'git_status',
    }
    assert 'Explore Subagent' in client.calls[0]['system']
