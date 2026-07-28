'''Single-request model execution boundary for the Agent Loop.'''

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from forge.runtime.model_client import ModelClient
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)


@dataclass(slots=True)
class ModelRun:
    '''One observable model request plus its accumulated response.'''

    runner: ModelRunner
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    system: str
    completed_usage: TokenUsage
    iteration: int
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    request_usage: TokenUsage | None = None

    @property
    def text(self) -> str:
        return ''.join(self.text_parts).strip()

    async def __aiter__(self) -> AsyncIterator[ModelStreamEvent]:
        async for event in self.runner.stream(
            messages=self.messages,
            tools=self.tools,
            system=self.system,
        ):
            if isinstance(event, ModelTextDelta):
                self.text_parts.append(event.text)
            elif isinstance(event, ModelToolCallCompleted):
                self.tool_calls.append(event.tool_call)
            elif isinstance(event, ModelUsageUpdate):
                self.request_usage = event.usage
                yield ModelUsageUpdate(
                    usage=add_token_usage(
                        self.completed_usage,
                        self.request_usage,
                    ),
                    request_usage=self.request_usage,
                    model_calls=self.iteration,
                )
                continue
            yield event


class ModelRunner:
    '''Delegate provider streaming while keeping it out of orchestration.'''

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def run(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system: str,
        completed_usage: TokenUsage,
        iteration: int,
    ) -> ModelRun:
        return ModelRun(
            runner=self,
            messages=messages,
            tools=tools,
            system=system,
            completed_usage=completed_usage,
            iteration=iteration,
        )

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system: str,
    ) -> AsyncIterator[ModelStreamEvent]:
        async for event in self.client.stream(
            messages=messages,
            tools=tools,
            system=system,
        ):
            yield event


def add_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_creation_input_tokens=(
            left.cache_creation_input_tokens
            + right.cache_creation_input_tokens
        ),
        cache_read_input_tokens=(
            left.cache_read_input_tokens + right.cache_read_input_tokens
        ),
    )
