'''Tests for the thin Anthropic SDK adapter.'''

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from forge.config import ForgeConfig
import forge.runtime.model_client as model_client_module
from forge.runtime.model_client import (
    AnthropicModelClient,
    ModelClient,
    ModelProtocolError,
)
from forge.runtime.state import (
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)


def usage(
    *,
    input_tokens: int | None,
    output_tokens: int,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


class FakeStream:
    def __init__(self, events: list[Any], final_message: Any) -> None:
        self.events = events
        self.final_message = final_message

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def __aiter__(self) -> Any:
        async def iterate() -> Any:
            for event in self.events:
                yield event

        return iterate()

    async def get_final_message(self) -> Any:
        return self.final_message


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[Any] = []
        self.final_message = SimpleNamespace(
            usage=usage(input_tokens=0, output_tokens=0)
        )

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.events, self.final_message)


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeMessages()


def test_client_passes_explicit_config_to_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeAnthropic()
    constructor_calls: list[dict[str, str]] = []

    def create_sdk(**kwargs: str) -> FakeAnthropic:
        constructor_calls.append(kwargs)
        return sdk

    monkeypatch.setattr(model_client_module, 'AsyncAnthropic', create_sdk)

    AnthropicModelClient(
        model='claude-test',
        config=ForgeConfig(
            api_key='test-api-key',
            model_id='claude-test',
            base_url='https://gateway.example.com/anthropic',
        ),
    )

    assert constructor_calls == [
        {
            'api_key': 'test-api-key',
            'base_url': 'https://gateway.example.com/anthropic',
        }
    ]


def test_client_can_use_model_id_from_config() -> None:
    sdk = FakeAnthropic()

    client = AnthropicModelClient.from_config(
        config=ForgeConfig(
            api_key='test-api-key',
            model_id='configured-model',
        ),
        client=sdk,
    )

    assert client.model == 'configured-model'


def collect_stream(
    client: AnthropicModelClient,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
) -> list[Any]:
    async def collect() -> list[Any]:
        return [
            event
            async for event in client.stream(
                messages=messages,
                tools=tools,
                system=system,
            )
        ]

    return asyncio.run(collect())


def test_stream_delegates_events_and_merges_usage() -> None:
    sdk = FakeAnthropic()
    sdk.messages.events = [
        SimpleNamespace(
            type='message_start',
            message=SimpleNamespace(
                usage=usage(input_tokens=100, output_tokens=0)
            ),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=0,
            delta=SimpleNamespace(type='text_delta', text='Hello'),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=0,
            delta=SimpleNamespace(type='text_delta', text=' world'),
        ),
        SimpleNamespace(
            type='message_delta',
            usage=usage(input_tokens=None, output_tokens=2),
        ),
    ]
    sdk.messages.final_message = SimpleNamespace(
        usage=usage(input_tokens=100, output_tokens=2)
    )
    client = AnthropicModelClient(
        model='claude-test',
        max_tokens=2048,
        client=sdk,
    )
    messages = [{'role': 'user', 'content': 'Read the repository.'}]
    tools = [
        {
            'name': 'read_file',
            'description': 'Read one repository file.',
            'input_schema': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        }
    ]

    events = collect_stream(
        client,
        messages=messages,
        tools=tools,
        system='You are a coding agent.',
    )

    assert events == [
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=100, output_tokens=0)
        ),
        ModelTextDelta(text='Hello'),
        ModelTextDelta(text=' world'),
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=100, output_tokens=2)
        ),
    ]
    assert sdk.messages.calls == [
        {
            'model': 'claude-test',
            'max_tokens': 2048,
            'messages': [
                {'role': 'user', 'content': 'Read the repository.'}
            ],
            'tools': [
                {
                    'name': 'read_file',
                    'description': 'Read one repository file.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {'path': {'type': 'string'}},
                        'required': ['path'],
                    },
                }
            ],
            'system': 'You are a coding agent.',
        }
    ]
    assert isinstance(client, ModelClient)


def test_adapter_passes_plain_message_dicts_to_sdk() -> None:
    sdk = FakeAnthropic()
    client = AnthropicModelClient(model='claude-test', client=sdk)
    messages = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'I will inspect it.'},
                {
                    'type': 'tool_use',
                    'id': 'toolu_read',
                    'name': 'read_file',
                    'input': {'path': 'README.md'},
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_read',
                    'content': 'file contents',
                    'is_error': False,
                }
            ],
        },
    ]

    collect_stream(client, messages=messages)

    assert sdk.messages.calls[0]['messages'] == messages


def test_stream_emits_multiple_completed_tool_calls() -> None:
    sdk = FakeAnthropic()
    sdk.messages.events = [
        SimpleNamespace(
            type='message_start',
            message=SimpleNamespace(
                usage=usage(input_tokens=80, output_tokens=0)
            ),
        ),
        SimpleNamespace(
            type='content_block_start',
            index=0,
            content_block=SimpleNamespace(
                type='tool_use',
                id='toolu_read',
                name='read_file',
                input={},
            ),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=0,
            delta=SimpleNamespace(
                type='input_json_delta',
                partial_json='{"path":',
            ),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=0,
            delta=SimpleNamespace(
                type='input_json_delta',
                partial_json='"README.md"}',
            ),
        ),
        SimpleNamespace(type='content_block_stop', index=0),
        SimpleNamespace(
            type='content_block_start',
            index=1,
            content_block=SimpleNamespace(
                type='tool_use',
                id='toolu_grep',
                name='grep',
                input={},
            ),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=1,
            delta=SimpleNamespace(
                type='input_json_delta',
                partial_json='{"pattern":"TODO","path":"forge"}',
            ),
        ),
        SimpleNamespace(type='content_block_stop', index=1),
        SimpleNamespace(
            type='message_delta',
            usage=usage(input_tokens=None, output_tokens=24),
        ),
    ]
    sdk.messages.final_message = SimpleNamespace(
        usage=usage(input_tokens=80, output_tokens=24)
    )
    client = AnthropicModelClient(model='claude-test', client=sdk)

    events = collect_stream(
        client,
        messages=[
            {'role': 'user', 'content': 'Inspect the repository.'}
        ],
    )

    assert events == [
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=80, output_tokens=0)
        ),
        ModelToolCallStarted(
            index=0,
            id='toolu_read',
            name='read_file',
        ),
        ModelToolCallArgumentsDelta(index=0, partial_json='{"path":'),
        ModelToolCallArgumentsDelta(
            index=0,
            partial_json='"README.md"}',
        ),
        ModelToolCallCompleted(
            tool_call=ToolCall(
                index=0,
                id='toolu_read',
                name='read_file',
                arguments={'path': 'README.md'},
            )
        ),
        ModelToolCallStarted(
            index=1,
            id='toolu_grep',
            name='grep',
        ),
        ModelToolCallArgumentsDelta(
            index=1,
            partial_json='{"pattern":"TODO","path":"forge"}',
        ),
        ModelToolCallCompleted(
            tool_call=ToolCall(
                index=1,
                id='toolu_grep',
                name='grep',
                arguments={'pattern': 'TODO', 'path': 'forge'},
            )
        ),
        ModelUsageUpdate(
            usage=TokenUsage(input_tokens=80, output_tokens=24)
        ),
    ]


def test_stream_rejects_invalid_tool_argument_json() -> None:
    sdk = FakeAnthropic()
    sdk.messages.events = [
        SimpleNamespace(
            type='content_block_start',
            index=0,
            content_block=SimpleNamespace(
                type='tool_use',
                id='toolu_bad',
                name='read_file',
                input={},
            ),
        ),
        SimpleNamespace(
            type='content_block_delta',
            index=0,
            delta=SimpleNamespace(
                type='input_json_delta',
                partial_json='["README.md"]',
            ),
        ),
        SimpleNamespace(type='content_block_stop', index=0),
    ]
    client = AnthropicModelClient(model='claude-test', client=sdk)

    with pytest.raises(ModelProtocolError, match='must be a JSON object'):
        collect_stream(
            client,
            messages=[{'role': 'user', 'content': 'Read a file.'}],
        )


def test_stream_omits_unused_optional_parameters() -> None:
    sdk = FakeAnthropic()
    client = AnthropicModelClient(model='claude-test', client=sdk)

    collect_stream(
        client,
        messages=[{'role': 'user', 'content': 'Hello'}],
    )

    call = sdk.messages.calls[0]
    assert 'tools' not in call
    assert 'system' not in call


@pytest.mark.parametrize(
    ('model', 'max_tokens'),
    [('', 1), ('claude-test', 0)],
)
def test_invalid_configuration_is_rejected(
    model: str,
    max_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        AnthropicModelClient(
            model=model,
            max_tokens=max_tokens,
            client=FakeAnthropic(),
        )
