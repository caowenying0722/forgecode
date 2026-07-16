'''Model client boundary backed by the official Anthropic SDK.'''

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json
from typing import Any, Protocol, runtime_checkable

from anthropic import AsyncAnthropic, omit
from anthropic.types import MessageParam, ToolParam

from forge.config import ForgeConfig
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelToolCallCompleted,
    ModelToolCallStarted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)


DEFAULT_MODEL_PROVIDER = 'anthropic'


class ModelProtocolError(RuntimeError):
    '''Raised when a provider emits an invalid model stream.'''


@dataclass(slots=True)
class _PendingToolCall:
    id: str
    name: str
    initial_input: dict[str, Any]
    json_parts: list[str] = field(default_factory=list)


@runtime_checkable
class ModelClient(Protocol):
    '''Minimal async interface used by the ForgeCode runtime.'''

    provider: str

    def stream(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        '''Stream text and exact provider usage updates.'''
        ...


class AnthropicModelClient:
    '''Thin adapter around Anthropic AsyncAnthropic.messages.stream.'''

    provider = DEFAULT_MODEL_PROVIDER

    @classmethod
    def from_config(
        cls,
        config: ForgeConfig | None = None,
        max_tokens: int = 4096,
        client: AsyncAnthropic | None = None,
    ) -> AnthropicModelClient:
        '''Create a model client from .env or an explicit ForgeConfig.'''
        resolved_config = config if config is not None else ForgeConfig.from_env()
        return cls(
            model=resolved_config.model_id,
            max_tokens=max_tokens,
            config=resolved_config,
            client=client,
        )

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        config: ForgeConfig | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if not model:
            raise ValueError('model must not be empty')
        if max_tokens < 1:
            raise ValueError('max_tokens must be positive')

        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            resolved_config = (
                config if config is not None else ForgeConfig.from_env()
            )
            self._client = AsyncAnthropic(
                api_key=resolved_config.api_key,
                base_url=resolved_config.base_url,
            )

    async def stream(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        current_usage: TokenUsage | None = None
        pending_tool_calls: dict[int, _PendingToolCall] = {}
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools if tools else omit,
            system=system if system is not None else omit,
        ) as stream:
            async for event in stream:
                if (
                    event.type == 'content_block_delta'
                    and event.delta.type == 'text_delta'
                ):
                    yield ModelTextDelta(
                        text=event.delta.text,
                        index=event.index,
                    )
                elif (
                    event.type == 'content_block_start'
                    and event.content_block.type == 'tool_use'
                ):
                    block = event.content_block
                    pending_tool_calls[event.index] = _PendingToolCall(
                        id=block.id,
                        name=block.name,
                        initial_input=dict(block.input),
                    )
                    yield ModelToolCallStarted(
                        index=event.index,
                        id=block.id,
                        name=block.name,
                    )
                elif (
                    event.type == 'content_block_delta'
                    and event.delta.type == 'input_json_delta'
                ):
                    pending = pending_tool_calls.get(event.index)
                    if pending is None:
                        raise ModelProtocolError(
                            'Received tool arguments before tool_use started '
                            f'at content block {event.index}.'
                        )
                    pending.json_parts.append(event.delta.partial_json)
                    yield ModelToolCallArgumentsDelta(
                        index=event.index,
                        partial_json=event.delta.partial_json,
                    )
                elif (
                    event.type == 'content_block_stop'
                    and event.index in pending_tool_calls
                ):
                    pending = pending_tool_calls.pop(event.index)
                    arguments = parse_tool_arguments(
                        pending,
                        index=event.index,
                    )
                    yield ModelToolCallCompleted(
                        tool_call=ToolCall(
                            index=event.index,
                            id=pending.id,
                            name=pending.name,
                            arguments=arguments,
                        )
                    )
                elif event.type == 'message_start':
                    current_usage = merge_usage(
                        event.message.usage,
                        current_usage,
                    )
                    yield ModelUsageUpdate(usage=current_usage)
                elif event.type == 'message_delta':
                    current_usage = merge_usage(
                        event.usage,
                        current_usage,
                    )
                    yield ModelUsageUpdate(usage=current_usage)

            final_message = await stream.get_final_message()

        if pending_tool_calls:
            indexes = ', '.join(str(index) for index in pending_tool_calls)
            raise ModelProtocolError(
                f'Tool calls did not finish at content blocks: {indexes}.'
            )

        final_usage = merge_usage(final_message.usage, current_usage)
        if final_usage != current_usage:
            yield ModelUsageUpdate(usage=final_usage)


def parse_tool_arguments(
    pending: _PendingToolCall,
    *,
    index: int,
) -> dict[str, Any]:
    '''Parse and validate one completed tool call JSON object.'''
    raw_json = ''.join(pending.json_parts)
    if not raw_json:
        return pending.initial_input

    try:
        arguments = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise ModelProtocolError(
            f'Invalid JSON arguments for tool {pending.name!r} '
            f'at content block {index}: {error.msg}.'
        ) from error
    if not isinstance(arguments, dict):
        raise ModelProtocolError(
            f'Arguments for tool {pending.name!r} at content block '
            f'{index} must be a JSON object.'
        )
    return arguments


def merge_usage(
    usage: Any,
    previous: TokenUsage | None = None,
) -> TokenUsage:
    '''Merge partial streaming usage fields into one exact snapshot.'''
    fallback = previous or TokenUsage(input_tokens=0, output_tokens=0)

    def value(name: str, default: int) -> int:
        reported = getattr(usage, name, None)
        return default if reported is None else reported

    return TokenUsage(
        input_tokens=value('input_tokens', fallback.input_tokens),
        output_tokens=value('output_tokens', fallback.output_tokens),
        cache_creation_input_tokens=value(
            'cache_creation_input_tokens',
            fallback.cache_creation_input_tokens,
        ),
        cache_read_input_tokens=value(
            'cache_read_input_tokens',
            fallback.cache_read_input_tokens,
        ),
    )
