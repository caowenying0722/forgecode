'''Runtime value objects shared across the Agent Loop and terminal UI.'''

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    '''One text fragment emitted by a streaming model response.'''

    text: str


@dataclass(frozen=True, slots=True)
class ModelUsageUpdate:
    '''Latest exact usage reported by the model provider.'''

    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    '''Final validated result for one streamed conversation turn.'''

    result: TurnResult


type ModelStreamEvent = ModelTextDelta | ModelUsageUpdate
type ConversationEvent = ModelStreamEvent | TurnCompleted
