'''Runtime value objects shared across the Agent Loop and terminal UI.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from forge.runtime.agent_state import AgentPhase
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
class VerificationEvidence:
    '''One verify result tied to an exact source revision.'''

    command: str
    cwd: str
    exit_code: int
    duration_seconds: float
    timed_out: bool
    workspace_revision: int
    status: str = 'passed'
    command_id: str = ''
    failure_signature: str = ''
    source_revision: int | None = None
    filesystem_revision: int | None = None
    verification_type: str = 'auto'
    verification_reused: bool = False
    generated_artifact_paths: tuple[str, ...] = ()
    cache_paths: tuple[str, ...] = ()
    verification_side_effect_paths: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.status == 'passed' and not self.timed_out and self.exit_code == 0

    @property
    def bound_source_revision(self) -> int:
        return self.workspace_revision if self.source_revision is None else self.source_revision


TaskStatus = Literal['completed', 'blocked', 'stuck', 'failed']


@dataclass(frozen=True, slots=True)
class TurnResult:
    '''Displayable response, evidence, and usage for one conversation turn.'''

    text: str
    usage: TokenUsage
    last_request_usage: TokenUsage | None = None
    model_calls: int = 1
    tool_calls: tuple[ToolCall, ...] = ()
    status: TaskStatus = 'completed'
    changed_paths: tuple[str, ...] = ()
    verification: VerificationEvidence | None = None
    completion_reasons: tuple[str, ...] = ()


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
    '''Latest request usage plus cumulative usage for the current turn.'''

    usage: TokenUsage
    request_usage: TokenUsage | None = None
    model_calls: int = 1


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted:
    '''Provider-level completion metadata for one streamed response.'''

    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class ModelCallStarted:
    '''One model request started inside the current user turn.'''

    iteration: int


@dataclass(frozen=True, slots=True)
class ModelRetryScheduled:
    '''A transient provider failure will be retried after a delay.'''

    attempt: int
    reason: str
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class ModelCallCompleted:
    '''One model request completed successfully.'''

    iteration: int


@dataclass(frozen=True, slots=True)
class ModelCallFailed:
    '''One model request ended without a usable response.'''

    iteration: int
    reason: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class AgentPhaseChanged:
    '''The orchestration state machine entered a new phase.'''

    phase: AgentPhase
    previous_phase: AgentPhase | None
    reason: str
    iteration: int


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
class WorkspaceChanged:
    '''A tool changed repository content during the current turn.'''

    revision: int
    paths: tuple[str, ...]
    filesystem_revision: int | None = None
    source_revision: int | None = None
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationCompleted:
    '''A verify tool produced evidence for one workspace revision.'''

    evidence: VerificationEvidence


@dataclass(frozen=True, slots=True)
class AcceptanceLedgerUpdated:
    '''Acceptance evidence changed for one or more task criteria.'''

    evidence: tuple[dict[str, Any], ...]
    completed_criterion_ids: tuple[str, ...] = ()
    source_revision: int = 0
    ledger: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionBlocked:
    '''The runtime rejected a premature model completion.'''

    attempt: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextCompacted:
    '''The runtime summarized conversation history before continuing.'''

    before_characters: int
    after_characters: int
    transcript_path: str | None
    automatic: bool = True


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
    | ModelResponseCompleted
    | ModelRetryScheduled
)
type ConversationEvent = (
    ModelStreamEvent
    | ModelCallStarted
    | ModelCallCompleted
    | ModelCallFailed
    | AgentPhaseChanged
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | WorkspaceChanged
    | VerificationCompleted
    | AcceptanceLedgerUpdated
    | CompletionBlocked
    | ContextCompacted
    | TurnCompleted
)
