'''Bounded read-only subagents exposed as ForgeCode tools.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from forge.tools.base import Tool, ToolInput, ToolRegistry, ToolResult
from forge.tools.filesystem import ListDirectoryTool, ReadFileTool
from forge.tools.git import GitStatusTool
from forge.tools.search import FindFilesTool, GrepTool


EXPLORE_SUBAGENT_SYSTEM = '''You are ForgeCode Explore Subagent.
You perform bounded, read-only repository exploration for the main agent.
Use only the provided read-only tools. Do not modify files, run tests, or
claim completion of the user task. Return a concise structured report with:
- relevant_files
- evidence
- root_cause_hypotheses
- suggested_edit_points
- open_questions
Ground every claim in observed repository evidence.'''


class SubagentModelClient(Protocol):
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> Any:
        ...


class ExploreSubagentInput(ToolInput):
    task: str = Field(min_length=1, max_length=2_000)
    focus_paths: list[str] = Field(default_factory=list, max_length=10)
    max_rounds: int = Field(default=4, ge=1, le=6)


class ExploreSubagentTool(Tool[ExploreSubagentInput]):
    name = 'explore_subagent'
    description = (
        'Delegate bounded read-only repository exploration to an isolated '
        'subagent. Use this to locate relevant files, gather evidence, and '
        'form hypotheses before editing. The subagent cannot write files or '
        'run process commands; it returns a structured report for the main '
        'agent to decide how to proceed.'
    )
    input_model = ExploreSubagentInput

    def __init__(
        self,
        root: Path,
        *,
        client: SubagentModelClient | None = None,
    ) -> None:
        super().__init__(root)
        self.client = client

    async def execute(self, arguments: ExploreSubagentInput) -> ToolResult:
        from forge.runtime.model_client import AnthropicModelClient

        client = self.client or AnthropicModelClient.from_config()
        subagent = ExploreSubagent(self.root, client)
        return await subagent.run(arguments)


class ExploreSubagent:
    '''A small isolated model loop with only read-only repository tools.'''

    def __init__(self, root: Path, client: SubagentModelClient) -> None:
        self.root = root
        self.client = client
        self.registry = ToolRegistry(
            [
                ListDirectoryTool(root),
                FindFilesTool(root),
                ReadFileTool(root),
                GrepTool(root),
                GitStatusTool(root),
            ]
        )

    async def run(self, arguments: ExploreSubagentInput) -> ToolResult:
        from forge.runtime.state import (
            ModelTextDelta,
            ModelToolCallCompleted,
            ModelUsageUpdate,
            TokenUsage,
        )

        messages: list[dict[str, Any]] = [
            {
                'role': 'user',
                'content': render_explore_task(arguments),
            }
        ]
        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        tool_calls: list[str] = []
        final_text = ''

        for round_index in range(1, arguments.max_rounds + 1):
            text_parts: list[str] = []
            requested: list[Any] = []
            request_usage: TokenUsage | None = None
            async for event in self.client.stream(
                messages,
                tools=self.registry.definitions,
                system=EXPLORE_SUBAGENT_SYSTEM,
            ):
                if isinstance(event, ModelTextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ModelToolCallCompleted):
                    requested.append(event.tool_call)
                elif isinstance(event, ModelUsageUpdate):
                    request_usage = event.usage
            if request_usage is not None:
                total_usage = add_usage(total_usage, request_usage)
            text = ''.join(text_parts).strip()
            if not requested:
                final_text = text
                break
            messages.append(build_assistant_message(text, requested))
            results: list[tuple[ToolCall, ToolResult]] = []
            for tool_call in requested:
                result = await self.registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                )
                results.append((tool_call, result))
                tool_calls.append(tool_call.name)
            messages.append(build_tool_result_message(results))
            if round_index == arguments.max_rounds:
                messages.append(
                    {
                        'role': 'user',
                        'content': (
                            'Round limit reached. Return the structured '
                            'exploration report now using existing evidence.'
                        ),
                    }
                )

        if not final_text:
            final_text = 'Explore subagent reached its round limit without a report.'
            return ToolResult.fail(
                'subagent_no_report',
                final_text,
                metadata=metadata(total_usage, tool_calls),
            )
        return ToolResult.ok(
            'Explore subagent returned a read-only report.',
            content=final_text,
            metadata=metadata(total_usage, tool_calls),
        )


def render_explore_task(arguments: ExploreSubagentInput) -> str:
    focus = (
        '\nFocus paths:\n' + '\n'.join(f'- {path}' for path in arguments.focus_paths)
        if arguments.focus_paths
        else ''
    )
    return f'Task:\n{arguments.task}{focus}'


def build_assistant_message(
    text: str,
    tool_calls: list[Any],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({'type': 'text', 'text': text})
    content.extend(
        {
            'type': 'tool_use',
            'id': call.id,
            'name': call.name,
            'input': call.arguments,
        }
        for call in sorted(tool_calls, key=lambda item: item.index)
    )
    return {'role': 'assistant', 'content': content}


def build_tool_result_message(
    results: list[tuple[Any, ToolResult]],
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': [
            {
                'type': 'tool_result',
                'tool_use_id': call.id,
                'is_error': not result.success,
                'content': json.dumps(
                    {
                        'success': result.success,
                        'summary': result.summary,
                        'content': result.content,
                        'error': (
                            None
                            if result.error is None
                            else {
                                'code': result.error.code,
                                'message': result.error.message,
                                'details': result.error.details,
                            }
                        ),
                        'metadata': result.metadata,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
            for call, result in results
        ],
    }


def add_usage(left: Any, right: Any) -> Any:
    from forge.runtime.state import TokenUsage

    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_creation_input_tokens=(
            left.cache_creation_input_tokens + right.cache_creation_input_tokens
        ),
        cache_read_input_tokens=(
            left.cache_read_input_tokens + right.cache_read_input_tokens
        ),
    )


def metadata(usage: Any, tool_calls: list[str]) -> dict[str, Any]:
    return {
        'subagent': 'explore',
        'input_tokens': usage.total_input_tokens,
        'output_tokens': usage.output_tokens,
        'tool_calls': tool_calls,
    }
