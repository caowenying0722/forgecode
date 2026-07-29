'''Deterministic control-plane state for the Agent Loop.'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from forge.runtime.intent import InitialToolSurface, TaskContract
from forge.runtime.recovery_manager import RepairTarget
from forge.runtime.state import VerificationEvidence
from forge.tools.base import ToolResult


class AgentControlState(str, Enum):
    INIT = 'init'
    TASK_PLANNING = 'task_planning'
    EXPLORING = 'exploring'
    TARGETED_ANALYSIS = 'targeted_analysis'
    PLANNING = 'planning'
    IMPLEMENTING = 'implementing'
    FIX_REQUIRED = 'fix_required'
    READY_TO_VERIFY = 'ready_to_verify'
    VERIFYING = 'verifying'
    RECOVERING = 'recovering'
    DONE = 'done'
    BLOCKED = 'blocked'


@dataclass(slots=True)
class AgentController:
    '''Map structured tool failures onto bounded runtime state transitions.'''

    state: AgentControlState = AgentControlState.INIT
    contract: TaskContract | None = None
    planning_recovery_calls: int = 0
    runtime: TurnRuntimeState | None = None

    def begin_turn(self, contract: TaskContract) -> None:
        self.contract = contract
        self.state = initial_state(contract)
        self.planning_recovery_calls = 0
        self.runtime = TurnRuntimeState(
            control_state=self.state,
            contract=contract,
            requires_planning=contract.requires_plan,
        )

    def snapshot(self) -> 'TurnRuntimeState':
        if self.runtime is None:
            contract = self.contract
            if contract is None:
                raise RuntimeError('AgentController has no active turn.')
            self.runtime = TurnRuntimeState(
                control_state=self.state,
                contract=contract,
            )
        self.runtime.control_state = self.state
        self.runtime.planning_recovery_calls = self.planning_recovery_calls
        return self.runtime

    @property
    def planning_recovery(self) -> bool:
        '''Compatibility view; TASK_PLANNING is the source of truth.'''
        return self.state is AgentControlState.TASK_PLANNING

    @property
    def action_recovery(self) -> bool:
        '''Compatibility view; TARGETED_ANALYSIS is the source of truth.'''
        return self.state is AgentControlState.TARGETED_ANALYSIS

    def initial_tool_surface(self) -> InitialToolSurface:
        if self.contract is None:
            return 'all'
        return self.contract.initial_tool_surface

    def enter_planning_recovery(self) -> None:
        self.state = AgentControlState.TASK_PLANNING
        self.planning_recovery_calls += 1
        if self.runtime is not None:
            self.runtime.control_state = self.state
            self.runtime.planning_recovery_calls = self.planning_recovery_calls

    def enter_implementing(self) -> None:
        self.state = AgentControlState.IMPLEMENTING
        if self.runtime is not None:
            self.runtime.control_state = self.state

    def enter_targeted_analysis(self) -> None:
        self.state = AgentControlState.TARGETED_ANALYSIS
        if self.runtime is not None:
            self.runtime.control_state = self.state

    def enter_fix_required(self) -> None:
        self.state = AgentControlState.FIX_REQUIRED
        if self.runtime is not None:
            self.runtime.control_state = self.state

    def enter_ready_to_verify(self) -> None:
        self.state = AgentControlState.READY_TO_VERIFY
        if self.runtime is not None:
            self.runtime.control_state = self.state

    def observe_tool_result(self, tool_name: str, result: ToolResult) -> None:
        if tool_name == 'todo_write' and result.success:
            self.state = AgentControlState.IMPLEMENTING
        elif tool_name == 'verify' and result.success:
            self.state = AgentControlState.VERIFYING
        elif tool_name == 'verify' and not result.success:
            self.enter_fix_required()


def initial_state(contract: TaskContract) -> AgentControlState:
    if contract.requires_plan:
        return AgentControlState.PLANNING
    if contract.initial_phase == 'planning':
        return AgentControlState.PLANNING
    if contract.initial_phase == 'implementing':
        return AgentControlState.IMPLEMENTING
    return AgentControlState.EXPLORING


@dataclass(slots=True)
class BudgetLedger:
    '''Turn-scoped budget accounting owned by controller state.'''

    model_calls: int = 0
    tool_calls: int = 0
    read_calls: int = 0
    write_calls: int = 0
    planning_calls: int = 0
    verification_calls: int = 0
    recovery_calls: int = 0
    max_model_calls: int | None = None
    max_tool_calls: int | None = None

    def observe_model_call(self) -> None:
        self.model_calls += 1

    def observe_tool_call(self, effect: str | None, tool_name: str) -> None:
        self.tool_calls += 1
        if effect == 'read_only':
            self.read_calls += 1
        elif effect == 'workspace_write':
            self.write_calls += 1
        if tool_name == 'todo_write':
            self.planning_calls += 1
        elif tool_name == 'verify':
            self.verification_calls += 1

    def exceeded(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if (
            self.max_model_calls is not None
            and self.model_calls > self.max_model_calls
        ):
            reasons.append(
                f'model call budget exceeded: '
                f'{self.model_calls}/{self.max_model_calls}'
            )
        if (
            self.max_tool_calls is not None
            and self.tool_calls > self.max_tool_calls
        ):
            reasons.append(
                f'tool call budget exceeded: '
                f'{self.tool_calls}/{self.max_tool_calls}'
            )
        return tuple(reasons)


@dataclass(slots=True)
class TurnRuntimeState:
    '''Single controller-owned snapshot for one active user turn.'''

    control_state: AgentControlState
    contract: TaskContract
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    current_step: str = ''
    failure_reasons: tuple[str, ...] = ()
    repair_target: RepairTarget | None = None
    latest_verification: VerificationEvidence | None = None
    verification_failed_revision: int | None = None
    verification_repair_revision: int | None = None
    verification_read_count: int = 0
    planning_recovery_calls: int = 0
    requires_planning: bool = False
    mutation_recovery_context: str = ''
    mutation_failures: tuple[dict[str, object], ...] = ()
    mutation_read_used: bool = False
    completion_ready_context: str = ''
    action_recovery_calls: int = 0
    action_read_used: bool = False
    synthesis_mode: str = ''

    @property
    def verification_recovery(self) -> bool:
        if self.control_state in {
            AgentControlState.READY_TO_VERIFY,
        }:
            return True
        return (
            self.control_state is AgentControlState.FIX_REQUIRED
            and self.latest_verification is not None
        )

    @property
    def verification_fix_required(self) -> bool:
        return (
            self.control_state is AgentControlState.FIX_REQUIRED
            and self.latest_verification is not None
            and not self.latest_verification.success
        )


def is_todo_required_result(result: ToolResult) -> bool:
    return (
        result.error is not None
        and result.error.code == 'todo_required'
        and bool(result.metadata.get('todo_required'))
    )


def batch_requires_todo(results: list[tuple[object, ToolResult]]) -> bool:
    return bool(results) and any(
        is_todo_required_result(result) for _, result in results
    )


def build_planning_recovery_feedback(
    task_context: str,
    *,
    attempt: int,
) -> dict[str, str]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n'
            '[ForgeCode Planning Recovery]\n'
            'A write or process tool was rejected because this complex task '
            'requires an explicit TODO plan first. The next request exposes '
            'only todo_write. Call todo_write with a short working plan and '
            'exactly one in_progress item, then continue implementation in '
            'the following request. Do not retry write/process tools before '
            f'todo_write succeeds. Planning recovery attempt: {attempt}.'
        ),
    }
