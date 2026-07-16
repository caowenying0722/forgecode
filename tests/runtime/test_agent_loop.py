'''Tests for the minimal M1 streaming conversation runtime.'''

import asyncio
from collections.abc import AsyncIterator

import pytest
from anthropic.types import MessageParam, ToolParam

from forge.runtime.agent_loop import (
    Conversation,
    ModelResponseError,
    load_system_prompt,
)
from forge.runtime.state import (
    ConversationEvent,
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsageUpdate,
    TokenUsage,
    TurnCompleted,
    TurnResult,
)


class FakeModelClient:
    '''Record requests and emit deterministic model stream events.'''

    provider = 'fake'

    def __init__(self, *responses: list[ModelStreamEvent]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def stream(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
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


def test_conversation_forwards_stream_and_returns_final_result() -> None:
    client = FakeModelClient(streamed_response('RE', 'ADY'))
    conversation = Conversation(client=client)

    events = collect_turn(conversation, 'Only reply READY')

    assert events == [
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=10, output_tokens=0)
        ),
        ModelTextDelta(text='RE'),
        ModelTextDelta(text='ADY'),
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=10, output_tokens=2)
        ),
        TurnCompleted(
            result=TurnResult(
                text='READY',
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            )
        ),
    ]
    assert client.calls[0]['messages'] == [
        {'role': 'user', 'content': 'Only reply READY'},
    ]
    assert client.calls[0]['tools'] is None
    assert client.calls[0]['system'] == load_system_prompt()


def test_system_prompt_defines_forgecode_identity() -> None:
    prompt = load_system_prompt()

    assert 'Your product identity is ForgeCode.' in prompt
    assert 'Do not claim to be Anthropic' in prompt
    assert 'This M1.1 runtime supports conversation only.' in prompt


def test_conversation_accepts_an_explicit_system_prompt() -> None:
    client = FakeModelClient(streamed_response('READY'))
    conversation = Conversation(
        client=client,
        system_prompt='test system',
    )

    collect_turn(conversation, 'hello')

    assert client.calls[0]['system'] == 'test system'


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
