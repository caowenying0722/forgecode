'''Tests for the minimal M1 model conversation runtime.'''

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from anthropic.types import Message, MessageParam, ToolParam

from forge.runtime.agent_loop import (
    Conversation,
    ModelResponseError,
    extract_text,
    load_system_prompt,
    run_single_turn,
)


def message_with_blocks(*blocks: object) -> Message:
    '''Build the small response shape needed by these unit tests.'''
    return cast(Message, SimpleNamespace(content=list(blocks)))


class FakeModelClient:
    '''Record requests and return a deterministic model response.'''

    provider = 'fake'

    def __init__(self, *responses: Message) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        system: str | None = None,
    ) -> Message:
        self.calls.append(
            {'messages': messages, 'tools': tools, 'system': system}
        )
        return self.responses.pop(0)


def test_run_single_turn_sends_one_user_message() -> None:
    response = message_with_blocks(SimpleNamespace(type='text', text='READY'))
    client = FakeModelClient(response)

    result = asyncio.run(run_single_turn('Only reply READY', client=client))

    assert result == 'READY'
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
    response = message_with_blocks(SimpleNamespace(type='text', text='READY'))
    client = FakeModelClient(response)
    conversation = Conversation(client=client, system_prompt='test system')

    asyncio.run(conversation.send('hello'))

    assert client.calls[0]['system'] == 'test system'


def test_conversation_sends_previous_turns_as_context() -> None:
    client = FakeModelClient(
        message_with_blocks(SimpleNamespace(type='text', text='Hello')),
        message_with_blocks(
            SimpleNamespace(type='text', text='Your name is Ada')
        ),
    )
    conversation = Conversation(client=client)

    first = asyncio.run(conversation.send('Hello'))
    second = asyncio.run(conversation.send('What is my name?'))

    assert first == 'Hello'
    assert second == 'Your name is Ada'
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


def test_conversation_does_not_commit_an_invalid_response() -> None:
    client = FakeModelClient(
        message_with_blocks(SimpleNamespace(type='tool_use', name='search'))
    )
    conversation = Conversation(client=client)

    with pytest.raises(ModelResponseError):
        asyncio.run(conversation.send('Hello'))

    assert conversation.messages == []


def test_extract_text_joins_text_blocks_and_ignores_other_blocks() -> None:
    message = message_with_blocks(
        SimpleNamespace(type='thinking', thinking='private reasoning'),
        SimpleNamespace(type='text', text='first'),
        SimpleNamespace(type='text', text='second'),
    )

    assert extract_text(message) == 'first\nsecond'


def test_extract_text_rejects_response_without_text() -> None:
    message = message_with_blocks(
        SimpleNamespace(type='tool_use', name='read_file')
    )

    with pytest.raises(
        ModelResponseError,
        match='did not contain any text',
    ):
        extract_text(message)


def test_run_single_turn_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match='prompt must not be empty'):
        asyncio.run(run_single_turn('   '))
