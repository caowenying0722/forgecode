'''Runtime value objects shared across the Agent Loop and terminal UI.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.tools.base import ToolResult


@dataclass(frozen=True, slots=True)
class TokenUsage:
    '''Token counts reported by one model request.'''

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            'input_tokens',
            'output_tokens',
            'cache_creation_input_tokens',
            'cache_read_input_tokens',
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f'{field_name} must not be negative')

    @property
    def total_input_tokens(self) -> int:
        '''Return regular, cache-write, and cache-read input tokens.'''
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        '''Return all input and output tokens processed for the request.'''
        return self.total_input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TurnResult:
    '''Displayable response and usage for one conversation turn.'''

    text: str
    usage: TokenUsage
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCall:
    '''One validated tool request produced by the model.'''

    index: int
    id: str
    name: str
    arguments: dict[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError('index must not be negative')
        if not self.id:
            raise ValueError('id must not be empty')
        if not self.name:
            raise ValueError('name must not be empty')


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    '''One text fragment emitted by a streaming model response.'''

    text: str
    index: int = 0


@dataclass(frozen=True, slots=True)
class ModelToolCallStarted:
    '''A model started streaming one tool request.'''

    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ModelToolCallArgumentsDelta:
    '''One partial JSON fragment for a streaming tool request.'''

    index: int
    partial_json: str


@dataclass(frozen=True, slots=True)
class ModelToolCallCompleted:
    '''A tool request has complete, validated JSON arguments.'''

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ModelUsageUpdate:
    '''Latest exact usage reported by the model provider.'''

    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    '''The runtime started executing one completed model tool request.'''

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolExecutionCompleted:
    '''The runtime finished one tool request with a structured result.'''

    tool_call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    '''Final validated result for one streamed conversation turn.'''

    result: TurnResult


type ModelStreamEvent = (
    ModelTextDelta
    | ModelToolCallStarted
    | ModelToolCallArgumentsDelta
    | ModelToolCallCompleted
    | ModelUsageUpdate
)
type ConversationEvent = (
    ModelStreamEvent
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | TurnCompleted
)
