'''Deterministic control-plane state for the Agent Loop.'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from forge.runtime.acceptance import AcceptanceLedger
from forge.runtime.intent import InitialToolSurface, TaskContract
from forge.runtime.recovery_manager import RepairTarget
from forge.runtime.state import VerificationEvidence
from forge.runtime.verification import (
    tool_result_verification_status,
    verification_status_requires_repair,
)
from forge.runtime.verification_ledger import VerificationLedger
from forge.tools.base import ToolResult


class AgentControlState(str, Enum):
    INIT = 'init'
    UNDERSTANDING = 'understanding'
    ANSWERING = 'answering'
    ADVISING = 'advising'
    TASK_PLANNING = 'task_planning'
    EXPLORING = 'exploring'
    TARGETED_ANALYSIS = 'targeted_analysis'
    DIAGNOSING = 'diagnosing'
    PLANNING = 'planning'
    IMPLEMENTING = 'implementing'
    FIX_REQUIRED = 'fix_required'
    READY_TO_VERIFY = 'ready_to_verify'
    VERIFYING = 'verifying'
    RECOVERING = 'recovering'
    FINALIZING = 'finalizing'
    DONE = 'done'
    BLOCKED = 'blocked'


class SynthesisMode(str, Enum):
    NORMAL = ''
    CHECKPOINT = 'checkpoint'
    FINALIZATION = 'finalization'
    STAGNATION_FINAL = 'stagnation_final'
    TOKEN_LIMIT = 'token_limit'


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
            status = tool_result_verification_status(
                result.metadata,
                success=result.success,
            )
            if verification_status_requires_repair(status):
                self.enter_fix_required()
            else:
                self.enter_ready_to_verify()


def initial_state(contract: TaskContract) -> AgentControlState:
    if contract.kind == 'answer':
        return AgentControlState.ANSWERING
    if contract.kind in {'advisory', 'status'}:
        return AgentControlState.ADVISING
    if contract.kind == 'inspect':
        return AgentControlState.EXPLORING
    if contract.kind == 'verify':
        return AgentControlState.VERIFYING
    if contract.requires_plan and contract.requires_change:
        return AgentControlState.TASK_PLANNING
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
class VerificationRecoveryState:
    latest: VerificationEvidence | None = None
    repair_target: RepairTarget | None = None
    failed_revision: int | None = None
    repair_revision: int | None = None
    read_count: int = 0
    recovery_calls: int = 0
    last_failure_signature: str = ''

    def clear(self) -> None:
        self.latest = None
        self.repair_target = None
        self.failed_revision = None
        self.repair_revision = None
        self.read_count = 0
        self.recovery_calls = 0
        self.last_failure_signature = ''

    def clear_failure(self) -> None:
        self.failed_revision = None
        self.repair_target = None
        self.read_count = 0
        self.recovery_calls = 0
        self.last_failure_signature = ''

    def requires_repair(self, control_state: AgentControlState) -> bool:
        return (
            control_state is AgentControlState.FIX_REQUIRED
            and self.latest is not None
            and verification_status_requires_repair(self.latest.status)
        )

    def recovery_active(self, control_state: AgentControlState) -> bool:
        if control_state is AgentControlState.READY_TO_VERIFY:
            return True
        return self.requires_repair(control_state)


@dataclass(slots=True)
class EditRecoveryState:
    context: str = ''
    failures: list[dict[str, object]] = field(default_factory=list)
    failure_count: int = 0
    read_used: bool = False

    @property
    def active(self) -> bool:
        return bool(self.failures)

    def clear(self) -> None:
        self.context = ''
        self.failures.clear()
        self.failure_count = 0
        self.read_used = False


@dataclass(slots=True)
class ActionRecoveryRuntimeState:
    calls: int = 0
    read_used: bool = False
    block_events: int = 0

    def reset(self) -> None:
        self.calls = 0
        self.read_used = False


@dataclass(slots=True)
class LoopRuntimeState:
    '''Counters that describe progress through one model-tool turn.'''

    calls_without_progress: int = 0
    pre_mutation_calls: int = 0
    tool_protocol_failures: int = 0

    def reset_progress(self) -> None:
        self.calls_without_progress = 0

    def reset_pre_mutation(self) -> None:
        self.pre_mutation_calls = 0

    def reset_protocol_failures(self) -> None:
        self.tool_protocol_failures = 0


@dataclass(slots=True)
class CompletionRuntimeState:
    '''Completion-gate counters and review coverage for one turn.'''

    blocks: int = 0
    last_reasons: tuple[str, ...] = ()
    ready_revision: int | None = None
    decision_calls: int = 0
    ready_context: str = ''
    reviewed_paths: set[str] = field(default_factory=set)

    def clear_ready(self) -> None:
        self.ready_revision = None
        self.decision_calls = 0
        self.ready_context = ''
        self.reviewed_paths.clear()


@dataclass(slots=True)
class ModelFailureRuntimeState:
    '''Model transport/protocol recovery counters for one turn.'''

    reactive_compaction_attempted: bool = False
    protocol_recoveries: int = 0
    output_continuations: int = 0

    def clear(self) -> None:
        self.reactive_compaction_attempted = False
        self.protocol_recoveries = 0
        self.output_continuations = 0


@dataclass(slots=True)
class SynthesisRuntimeState:
    mode: SynthesisMode = SynthesisMode.NORMAL
    retries: int = 0
    token_limit_reason: str = ''

    @property
    def force(self) -> bool:
        return self.mode is not SynthesisMode.NORMAL

    @force.setter
    def force(self, value: bool) -> None:
        if not value and self.mode is SynthesisMode.CHECKPOINT:
            self.mode = SynthesisMode.NORMAL
        elif value and self.mode is SynthesisMode.NORMAL:
            self.mode = SynthesisMode.CHECKPOINT

    def clear(self) -> None:
        self.mode = SynthesisMode.NORMAL
        self.retries = 0

    @property
    def finalization_recovery(self) -> bool:
        return self.mode is SynthesisMode.FINALIZATION

    @finalization_recovery.setter
    def finalization_recovery(self, value: bool) -> None:
        if value:
            self.mode = SynthesisMode.FINALIZATION
        elif self.mode is SynthesisMode.FINALIZATION:
            self.mode = SynthesisMode.NORMAL

    @property
    def stagnation_final_recovery(self) -> bool:
        return self.mode is SynthesisMode.STAGNATION_FINAL

    @stagnation_final_recovery.setter
    def stagnation_final_recovery(self, value: bool) -> None:
        if value:
            self.mode = SynthesisMode.STAGNATION_FINAL
        elif self.mode is SynthesisMode.STAGNATION_FINAL:
            self.mode = SynthesisMode.NORMAL

    @property
    def token_limit_recovery(self) -> bool:
        return self.mode is SynthesisMode.TOKEN_LIMIT

    @token_limit_recovery.setter
    def token_limit_recovery(self, value: bool) -> None:
        if value:
            self.mode = SynthesisMode.TOKEN_LIMIT
        elif self.mode is SynthesisMode.TOKEN_LIMIT:
            self.mode = SynthesisMode.NORMAL


@dataclass(slots=True)
class TurnRuntimeState:
    '''Single controller-owned snapshot for one active user turn.'''

    control_state: AgentControlState
    contract: TaskContract
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    current_step: str = ''
    failure_reasons: tuple[str, ...] = ()
    verification: VerificationRecoveryState = field(
        default_factory=VerificationRecoveryState
    )
    verification_ledger: VerificationLedger = field(
        default_factory=VerificationLedger
    )
    acceptance_ledger: AcceptanceLedger = field(
        default_factory=AcceptanceLedger
    )
    planning_recovery_calls: int = 0
    requires_planning: bool = False
    edit_recovery: EditRecoveryState = field(default_factory=EditRecoveryState)
    loop: LoopRuntimeState = field(default_factory=LoopRuntimeState)
    completion: CompletionRuntimeState = field(
        default_factory=CompletionRuntimeState
    )
    action_recovery_state: ActionRecoveryRuntimeState = field(
        default_factory=ActionRecoveryRuntimeState
    )
    model_failure: ModelFailureRuntimeState = field(
        default_factory=ModelFailureRuntimeState
    )
    synthesis: SynthesisRuntimeState = field(
        default_factory=SynthesisRuntimeState
    )

    @property
    def verification_recovery(self) -> bool:
        return self.verification.recovery_active(self.control_state)

    @property
    def verification_fix_required(self) -> bool:
        return self.verification.requires_repair(self.control_state)

    @property
    def latest_verification(self) -> VerificationEvidence | None:
        return self.verification.latest

    @latest_verification.setter
    def latest_verification(self, value: VerificationEvidence | None) -> None:
        self.verification.latest = value

    @property
    def repair_target(self) -> RepairTarget | None:
        return self.verification.repair_target

    @repair_target.setter
    def repair_target(self, value: RepairTarget | None) -> None:
        self.verification.repair_target = value

    @property
    def verification_failed_revision(self) -> int | None:
        return self.verification.failed_revision

    @verification_failed_revision.setter
    def verification_failed_revision(self, value: int | None) -> None:
        self.verification.failed_revision = value

    @property
    def verification_repair_revision(self) -> int | None:
        return self.verification.repair_revision

    @verification_repair_revision.setter
    def verification_repair_revision(self, value: int | None) -> None:
        self.verification.repair_revision = value

    @property
    def verification_read_count(self) -> int:
        return self.verification.read_count

    @verification_read_count.setter
    def verification_read_count(self, value: int) -> None:
        self.verification.read_count = value

    @property
    def mutation_recovery_context(self) -> str:
        return self.edit_recovery.context

    @mutation_recovery_context.setter
    def mutation_recovery_context(self, value: str) -> None:
        self.edit_recovery.context = value

    @property
    def mutation_failures(self) -> tuple[dict[str, object], ...]:
        return tuple(self.edit_recovery.failures)

    @mutation_failures.setter
    def mutation_failures(self, value: tuple[dict[str, object], ...]) -> None:
        self.edit_recovery.failures = list(value)

    @property
    def mutation_read_used(self) -> bool:
        return self.edit_recovery.read_used

    @mutation_read_used.setter
    def mutation_read_used(self, value: bool) -> None:
        self.edit_recovery.read_used = value

    @property
    def action_recovery_calls(self) -> int:
        return self.action_recovery_state.calls

    @action_recovery_calls.setter
    def action_recovery_calls(self, value: int) -> None:
        self.action_recovery_state.calls = value

    @property
    def action_read_used(self) -> bool:
        return self.action_recovery_state.read_used

    @action_read_used.setter
    def action_read_used(self, value: bool) -> None:
        self.action_recovery_state.read_used = value

    @property
    def completion_ready_context(self) -> str:
        return self.completion.ready_context

    @completion_ready_context.setter
    def completion_ready_context(self, value: str) -> None:
        self.completion.ready_context = value

    @property
    def synthesis_mode(self) -> str:
        return self.synthesis.mode.value

    @synthesis_mode.setter
    def synthesis_mode(self, value: str) -> None:
        self.synthesis.mode = SynthesisMode(value)


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
