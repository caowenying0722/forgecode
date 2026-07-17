'''Tests for the minimal M1 streaming conversation runtime.'''

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from forge.runtime.agent_loop import (
    AgentLoopLimitError,
    Conversation,
    ModelResponseError,
    load_system_prompt,
)
from forge.runtime.state import (
    ConversationEvent,
    ModelCallCompleted,
    ModelCallStarted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsageUpdate,
    TokenUsage,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnCompleted,
    TurnResult,
    ToolCall,
)
from forge.tools.base import Tool, ToolInput, ToolRegistry, ToolResult


class FakeModelClient:
    '''Record requests and emit deterministic model stream events.'''

    provider = 'fake'

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
            {'messages': messages, 'tools': tools, 'system': system}
        )
        for event in self.responses.pop(0):
            yield event


def streamed_response(
    *text_parts: str,
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> list[ModelStreamEvent]:
    return [
        ModelUsageUpdate(
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=0,
            )
        ),
        *(ModelTextDelta(text=part) for part in text_parts),
        ModelUsageUpdate(
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        ),
    ]


def collect_turn(
    conversation: Conversation,
    prompt: str,
) -> list[ConversationEvent]:
    async def collect() -> list[ConversationEvent]:
        return [event async for event in conversation.stream(prompt)]

    return asyncio.run(collect())


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1)


class RecordingReadFileTool(Tool[ReadFileInput]):
    name = 'read_file'
    description = 'Read a test file.'
    input_model = ReadFileInput

    def __init__(
        self,
        root: Path,
        result: ToolResult | None = None,
    ) -> None:
        super().__init__(root)
        self.calls: list[str] = []
        self.result = result or ToolResult.ok(
            'Read file.',
            content='file contents',
        )

    async def execute(self, arguments: ReadFileInput) -> ToolResult:
        self.calls.append(arguments.path)
        return self.result


def tool_response(
    *tool_calls: ToolCall,
    input_tokens: int = 15,
    output_tokens: int = 10,
) -> list[ModelStreamEvent]:
    events: list[ModelStreamEvent] = [
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=0)
        )
    ]
    for tool_call in tool_calls:
        events.extend(
            [
                ModelToolCallStarted(
                    index=tool_call.index,
                    id=tool_call.id,
                    name=tool_call.name,
                ),
                ModelToolCallArgumentsDelta(
                    index=tool_call.index,
                    partial_json=json.dumps(tool_call.arguments),
                ),
                ModelToolCallCompleted(tool_call=tool_call),
            ]
        )
    events.append(
        ModelUsageUpdate(
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
    )
    return events


def test_conversation_forwards_stream_and_returns_final_result() -> None:
    client = FakeModelClient(streamed_response('RE', 'ADY'))
    conversation = Conversation(client=client)

    events = collect_turn(conversation, 'Only reply READY')

    assert events == [
        ModelCallStarted(iteration=1),
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=10, output_tokens=0)
        ),
        ModelTextDelta(text='RE'),
        ModelTextDelta(text='ADY'),
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=10, output_tokens=2)
        ),
        ModelCallCompleted(iteration=1),
        TurnCompleted(
            result=TurnResult(
                text='READY',
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            )
        ),
    ]
    assert client.calls[0]['messages'] == [
        {'role': 'user', 'content': 'Only reply READY'}
    ]
    assert client.calls[0]['tools'] is None
    assert client.calls[0]['system'] == load_system_prompt()


def test_system_prompt_defines_forgecode_identity() -> None:
    prompt = load_system_prompt()

    assert 'Your product identity is ForgeCode.' in prompt
    assert 'Do not claim to be Anthropic' in prompt
    assert 'The M1.4 runtime can use built-in file' in prompt
    assert 'only when its schema is included' in prompt
    assert 'call another tool instead of giving a premature' in prompt
    assert 'Do not run destructive commands' in prompt


def test_conversation_accepts_an_explicit_system_prompt() -> None:
    client = FakeModelClient(streamed_response('READY'))
    conversation = Conversation(
        client=client,
        system_prompt='test system',
    )

    collect_turn(conversation, 'hello')

    assert client.calls[0]['system'] == 'test system'


def test_conversation_executes_tool_and_continues_until_final_text(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'README.md'},
    )
    client = FakeModelClient(
        tool_response(tool_call),
        streamed_response(
            'Finished',
            input_tokens=30,
            output_tokens=4,
        ),
    )
    tool = RecordingReadFileTool(tmp_path)
    registry = ToolRegistry([tool])
    conversation = Conversation(client=client, registry=registry)

    events = collect_turn(conversation, 'Read the README')

    assert ToolExecutionStarted(tool_call=tool_call) in events
    assert ToolExecutionCompleted(
        tool_call=tool_call,
        result=tool.result,
    ) in events
    assert events[-1] == TurnCompleted(
        result=TurnResult(
            text='Finished',
            usage=TokenUsage(input_tokens=45, output_tokens=14),
            tool_calls=(tool_call,),
        )
    )
    assert tool.calls == ['README.md']
    assert client.calls[0]['tools'] == registry.definitions
    second_request = client.calls[1]
    assert second_request['messages'][:2] == [
        {'role': 'user', 'content': 'Read the README'},
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'toolu_read',
                    'name': 'read_file',
                    'input': {'path': 'README.md'},
                }
            ],
        },
    ]
    tool_result_message = second_request['messages'][2]
    assert tool_result_message['role'] == 'user'
    assert len(tool_result_message['content']) == 1
    result_block = tool_result_message['content'][0]
    assert result_block['tool_use_id'] == 'toolu_read'
    assert result_block['is_error'] is False
    payload = json.loads(result_block['content'])
    assert payload == {
        'success': True,
        'summary': 'Read file.',
        'content': 'file contents',
        'error': None,
        'metadata': {},
    }
    assert conversation.messages == [
        {'role': 'user', 'content': 'Read the README'},
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'toolu_read',
                    'name': 'read_file',
                    'input': {'path': 'README.md'},
                }
            ],
        },
        tool_result_message,
        {'role': 'assistant', 'content': 'Finished'},
    ]


def test_conversation_executes_multiple_tool_calls_in_order(
    tmp_path: Path,
) -> None:
    first = ToolCall(
        index=0,
        id='toolu_first',
        name='read_file',
        arguments={'path': 'a.py'},
    )
    second = ToolCall(
        index=1,
        id='toolu_second',
        name='read_file',
        arguments={'path': 'b.py'},
    )
    client = FakeModelClient(
        tool_response(first, second),
        streamed_response('Done', input_tokens=25, output_tokens=3),
    )
    tool = RecordingReadFileTool(tmp_path)
    conversation = Conversation(
        client=client,
        registry=ToolRegistry([tool]),
    )

    events = collect_turn(conversation, 'Read both files')

    assert tool.calls == ['a.py', 'b.py']
    result_blocks = client.calls[1]['messages'][2]['content']
    assert [
        block['tool_use_id']
        for block in result_blocks
    ] == [
        'toolu_first',
        'toolu_second',
    ]
    assert events[-1].result.tool_calls == (first, second)


def test_failed_tool_result_is_returned_to_model(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall(
        index=0,
        id='toolu_missing',
        name='missing_tool',
        arguments={},
    )
    client = FakeModelClient(
        tool_response(tool_call),
        streamed_response('Could not run that tool.'),
    )
    conversation = Conversation(
        client=client,
        registry=ToolRegistry([RecordingReadFileTool(tmp_path)]),
    )

    collect_turn(conversation, 'Use a missing tool')

    blocks = client.calls[1]['messages'][2]['content']
    block = blocks[0]
    payload = json.loads(block['content'])
    assert block['is_error'] is True
    assert payload['success'] is False
    assert payload['error']['code'] == 'unknown_tool'


def test_agent_loop_stops_at_model_call_limit(tmp_path: Path) -> None:
    first = ToolCall(
        index=0,
        id='toolu_1',
        name='read_file',
        arguments={'path': 'a.py'},
    )
    second = ToolCall(
        index=0,
        id='toolu_2',
        name='read_file',
        arguments={'path': 'b.py'},
    )
    client = FakeModelClient(tool_response(first), tool_response(second))
    conversation = Conversation(
        client=client,
        registry=ToolRegistry([RecordingReadFileTool(tmp_path)]),
        max_iterations=2,
    )

    with pytest.raises(AgentLoopLimitError, match='exceeded 2'):
        collect_turn(conversation, 'Never finish')

    assert len(client.calls) == 2


def test_conversation_sends_previous_turns_as_context() -> None:
    client = FakeModelClient(
        streamed_response('Hello'),
        streamed_response('Your name is Ada', input_tokens=20),
    )
    conversation = Conversation(client=client)

    collect_turn(conversation, 'Hello')
    collect_turn(conversation, 'What is my name?')

    assert client.calls[1]['messages'] == [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hello'},
        {'role': 'user', 'content': 'What is my name?'},
    ]
    assert client.calls[1]['system'] == load_system_prompt()
    assert conversation.messages == [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hello'},
        {'role': 'user', 'content': 'What is my name?'},
        {'role': 'assistant', 'content': 'Your name is Ada'},
    ]


def test_conversation_does_not_commit_stream_without_text() -> None:
    client = FakeModelClient(
        [
            ModelUsageUpdate(
                usage=TokenUsage(input_tokens=10, output_tokens=0)
            )
        ]
    )
    conversation = Conversation(client=client)

    with pytest.raises(
        ModelResponseError,
        match='did not contain any text',
    ):
        collect_turn(conversation, 'Hello')

    assert conversation.messages == []


def test_conversation_does_not_commit_stream_without_usage() -> None:
    client = FakeModelClient([ModelTextDelta(text='Hello')])
    conversation = Conversation(client=client)

    with pytest.raises(
        ModelResponseError,
        match='did not contain token usage',
    ):
        collect_turn(conversation, 'Hello')

    assert conversation.messages == []


def test_conversation_rejects_empty_prompt() -> None:
    conversation = Conversation(client=FakeModelClient())

    with pytest.raises(ValueError, match='prompt must not be empty'):
        collect_turn(conversation, '   ')
