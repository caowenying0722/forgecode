'''Explicit state machine for one Agent Loop turn.'''

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class AgentPhase(StrEnum):
    '''Stable phases emitted while processing one user turn.'''

    THINKING = 'thinking'
    PREPARING_TOOLS = 'preparing_tools'
    EXECUTING_TOOLS = 'executing_tools'
    CHECKING_RESULT = 'checking_result'
    RECOVERING = 'recovering'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass(frozen=True, slots=True)
class AgentStateTransition:
    '''One explainable state change in an Agent Loop turn.'''

    previous: AgentPhase | None
    current: AgentPhase
    reason: str
    iteration: int
    occurred_at: datetime


@dataclass(slots=True)
class AgentRunState:
    '''Mutable state owned exclusively by the orchestration loop.'''

    phase: AgentPhase | None = None
    iteration: int = 0
    transitions: list[AgentStateTransition] = field(default_factory=list)

    def transition(
        self,
        phase: AgentPhase,
        *,
        reason: str,
        iteration: int,
    ) -> AgentStateTransition | None:
        if self.phase is phase:
            self.iteration = iteration
            return None
        transition = AgentStateTransition(
            previous=self.phase,
            current=phase,
            reason=reason,
            iteration=iteration,
            occurred_at=datetime.now(UTC),
        )
        self.phase = phase
        self.iteration = iteration
        self.transitions.append(transition)
        return transition
