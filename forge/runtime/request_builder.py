'''Build one model request from explicit Agent Loop state.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.runtime.agent_controller import AgentControlState, TurnRuntimeState
from forge.runtime.intent import TaskContract
from forge.runtime.recovery_feedback import render_action_recovery_context
from forge.runtime.recovery_manager import (
    RecoveryManager,
    RepairTarget,
    render_repair_target_context,
)
from forge.runtime.state import VerificationEvidence
from forge.runtime.task_model import (
    build_runtime_task_model,
    render_runtime_task_model,
)


@dataclass(frozen=True, slots=True)
class RequestState:
    '''Request inputs for one model call.

    ``runtime`` is the production source of truth. The scalar recovery fields
    remain as compatibility fallbacks for narrow tests and older callers that
    have not yet been migrated.
    '''

    runtime: TurnRuntimeState | None = None
    control_state: AgentControlState | None = None
    force_synthesis: bool = False
    mutation_recovery_context: str = ''
    mutation_failures: tuple[dict[str, Any], ...] = ()
    mutation_read_used: bool = False
    finalization_recovery: bool = False
    stagnation_final_recovery: bool = False
    token_limit_recovery: bool = False
    completion_ready_context: str = ''
    verification_recovery: bool = False
    verification_fix_recovery: bool = False
    verification_fix_required: bool = False
    verification_read_used: bool = False
    verification_read_count: int = 0
    latest_verification: VerificationEvidence | None = None
    verification_repair_target: RepairTarget | None = None
    planning_recovery: bool = False
    planning_recovery_calls: int = 0
    task_contract: TaskContract | None = None
    change_required: bool = False
    mutation_attempted: bool = False
    action_recovery: bool = False
    action_recovery_calls: int = 0
    action_read_used: bool = False
    task_scope_patterns: tuple[str, ...] = ()
    task_goal: str = ''

    @property
    def tool_free_recovery(self) -> bool:
        runtime = self.runtime
        if runtime is not None:
            return runtime.synthesis.mode.value in {
                'stagnation_final',
                'token_limit',
            }
        return (
            self.stagnation_final_recovery
            or self.token_limit_recovery
        )


@dataclass(frozen=True, slots=True)
class ModelRequestSpec:
    tools: list[dict[str, Any]] | None
    system_prompt: str

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(
            str(definition.get('name', ''))
            for definition in self.tools or ()
        )


class RequestBuilder:
    '''Own tool visibility and system-context assembly for one model call.'''

    def __init__(
        self,
        recovery_manager: RecoveryManager,
        *,
        action_recovery_limit: int,
    ) -> None:
        self.recovery_manager = recovery_manager
        self.action_recovery_limit = action_recovery_limit

    def build(
        self,
        *,
        state: RequestState,
        interaction_mode: str,
        all_tools: list[dict[str, Any]] | None,
        plan_tools: list[dict[str, Any]] | None,
        base_system_prompt: str,
        repository_context: str,
        changed_paths: tuple[str, ...],
    ) -> ModelRequestSpec:
        tools = self._select_tools(
            state,
            interaction_mode=interaction_mode,
            all_tools=all_tools,
            plan_tools=plan_tools,
        )
        prompt = self._system_prompt(
            base_system_prompt,
            repository_context=repository_context,
            changed_paths=changed_paths,
            state=state,
        )
        return ModelRequestSpec(tools=tools, system_prompt=prompt)

    def _select_tools(
        self,
        state: RequestState,
        *,
        interaction_mode: str,
        all_tools: list[dict[str, Any]] | None,
        plan_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        runtime = state.runtime
        control_state = _control_state(state)
        contract = _task_contract(state)
        if state.tool_free_recovery:
            return None
        if _synthesis_mode(state) == 'finalization' or (
            runtime is None and state.finalization_recovery
        ):
            return self.recovery_manager.finalization_tools()
        if _planning_recovery_active(state):
            return self.recovery_manager.planning_tools()
        mutation_failures = _mutation_failures(state)
        if (
            mutation_failures
            or (
                control_state is AgentControlState.FIX_REQUIRED
                and mutation_failures
            )
        ):
            mutation_read_used = _mutation_read_used(state)
            return self.recovery_manager.mutation_tools(
                list(mutation_failures),
                read_available=not mutation_read_used,
                include_finish=False,
            )
        if _verification_recovery_active(state):
            target = _verification_repair_target(state)
            read_budget = self.recovery_manager.verification_read_budget(
                target
            )
            read_count = _verification_read_count(state)
            if runtime is None and state.verification_read_used:
                read_count = max(read_count, read_budget)
            return self.recovery_manager.verification_tools(
                fix_available=_verification_fix_recovery_active(state),
                read_available=read_count < read_budget,
                verify_available=not _verification_fix_required(state),
            )
        if (
            control_state is AgentControlState.TARGETED_ANALYSIS
            or (control_state is None and state.action_recovery)
        ):
            action_read_used = _action_read_used(state)
            return self.recovery_manager.action_tools(
                read_available=not action_read_used
            )
        if (
            control_state is AgentControlState.PLANNING
            or interaction_mode == 'plan'
        ):
            return plan_tools
        if (
            contract is not None
            and contract.initial_tool_surface == 'read_only'
        ):
            return plan_tools
        if (
            contract is not None
            and contract.initial_tool_surface == 'none'
        ):
            return None
        return all_tools

    def _system_prompt(
        self,
        base: str,
        *,
        repository_context: str,
        changed_paths: tuple[str, ...],
        state: RequestState,
    ) -> str:
        prompt = base
        if repository_context:
            prompt += '\n\n' + repository_context
        contract = _task_contract(state)
        task_goal = _task_goal(state)
        task_scope_patterns = _task_scope_patterns(state)
        if contract is not None and task_goal:
            prompt += '\n\n' + render_runtime_task_model(
                build_runtime_task_model(
                    task_goal,
                    contract,
                    scope_patterns=task_scope_patterns,
                )
            )
        if _change_required(state):
            prompt += '\n\n' + render_change_contract_context(
                changed_paths,
                mutation_attempted=_mutation_attempted(state),
                contract=contract,
                task_scope_patterns=task_scope_patterns,
            )
        elif contract is not None:
            prompt += '\n\n' + render_task_contract_context(
                contract,
            )
        mutation_context = _mutation_recovery_context(state)
        if mutation_context:
            prompt += '\n\n' + mutation_context
        completion_context = _completion_ready_context(state)
        if completion_context:
            prompt += '\n\n' + completion_context
        repair_target_context = ''
        if (
            _verification_recovery_active(state)
            and _latest_verification(state) is not None
        ):
            target = _verification_repair_target(state)
            if target is None:
                target = self.recovery_manager.verification_repair_target(
                    _latest_verification(state),
                    changed_paths=changed_paths,
                )
            repair_target_context = render_repair_target_context(
                target
            )
        prompt += recovery_system_suffix(
            state,
            action_recovery_limit=self.action_recovery_limit,
            repair_target_context=repair_target_context,
        )
        return prompt


def render_change_contract_context(
    changed_paths: tuple[str, ...],
    *,
    mutation_attempted: bool,
    contract: TaskContract | None = None,
    task_scope_patterns: tuple[str, ...] = (),
) -> str:
    paths = ', '.join(changed_paths) if changed_paths else 'none'
    attempted = 'yes' if mutation_attempted else 'no'
    intent = render_contract_summary(contract) if contract is not None else ''
    scope = ''
    if task_scope_patterns:
        patterns = ', '.join(task_scope_patterns[:16])
        suffix = ' ...' if len(task_scope_patterns) > 16 else ''
        scope = (
            '\nTask-relevant path patterns: '
            f'{patterns}{suffix}\n'
            'Completion checks reject placeholder or temporary-only Diffs '
            'that do not match the task goal.'
        )
    return (
        '[ForgeCode Turn Change Contract]\n'
        f'{intent}'
        'The user requested an implemented workspace change; an explanation '
        'or inspection alone cannot complete this turn.\n'
        f'- task-local changed paths: {paths}\n'
        f'- workspace write attempted: {attempted}\n'
        'Only a file revision after the turn baseline satisfies this '
        'contract. Git HEAD changes or untracked files that already existed '
        'when the turn began are background context, not work completed in '
        f'this turn.{scope}'
    )


def render_task_contract_context(contract: TaskContract) -> str:
    return (
        '[ForgeCode Turn Task Contract]\n'
        f'{render_contract_summary(contract)}'
        'The initial tool surface follows this contract. If the user asked '
        'for planning, status, explanation, or inspection, do not perform '
        'workspace edits unless a later explicit user request changes the '
        'mode or objective.'
    )


def render_contract_summary(contract: TaskContract) -> str:
    return (
        f'- intent: {contract.intent.kind} '
        f'({contract.intent.confidence}, {contract.intent.reason})\n'
        f'- completion contract: {contract.completion_contract}\n'
        f'- initial phase: {contract.initial_phase}\n'
        f'- initial tool surface: {contract.initial_tool_surface}\n'
    )


def recovery_system_suffix(
    state: RequestState,
    *,
    action_recovery_limit: int,
    repair_target_context: str = '',
) -> str:
    synthesis_mode = _synthesis_mode(state)
    if synthesis_mode == 'finalization' or (
        state.runtime is None and state.finalization_recovery
    ):
        return (
            '\n\n[ForgeCode Finalization Recovery]\n'
            'The current workspace revision already has a real Diff and '
            'current successful verification. This is a dedicated final '
            'synthesis request. Return one concise final answer in the '
            "user's language or call finish_task alone. State what changed "
            'and the exact verification performed. Be honest about anything '
            'that was not semantically or visually verified. Do not request '
            'any tool except finish_task.'
        )
    if _planning_recovery_active(state):
        return (
            '\n\n[ForgeCode Planning Recovery]\n'
            'The previous tool call was rejected with TODO_REQUIRED. This '
            'request is restricted to todo_write. Call todo_write with a '
            'short working plan and exactly one in_progress item. Do not '
            'request any write, process, verification, discovery, or finish '
            'tool until todo_write succeeds. '
            f'Planning recovery count: {_planning_recovery_calls(state)}.'
        )
    if synthesis_mode == 'stagnation_final' or (
        state.runtime is None and state.stagnation_final_recovery
    ):
        return (
            '\n\n[ForgeCode Stagnation Final Recovery]\n'
            'The previous tool-enabled attempts did not produce new '
            'workspace, plan, or repository evidence. This is one '
            'dedicated final recovery request with no tools included. '
            "Return the best concise answer possible in the user's "
            'language using only the existing conversation and repository '
            'evidence. If the goal cannot be completed from the collected '
            'evidence, state the blocker and the most specific next '
            'action a future tool-enabled turn should take. Do not '
            'request or describe another tool call.'
        )
    if synthesis_mode == 'token_limit' or (
        state.runtime is None and state.token_limit_recovery
    ):
        return (
            '\n\n[ForgeCode Token-Limit Recovery]\n'
            'The current user turn reached its cumulative input-token '
            'safety threshold. This is one dedicated final recovery '
            'request with no tools included. Return a concise progress '
            "summary in the user's language using only existing "
            'conversation and repository evidence. State what was done, '
            'what remains, any verification already performed, and the '
            'specific next step a future turn should take. Do not request '
            'or describe another tool call.'
        )
    if _verification_recovery_active(state):
        verify_gate = (
            'A previous verification failed and no later workspace revision '
            'has been created yet, so verify is intentionally unavailable. '
            'Use the latest verification output to make a relevant repair '
            'first. After a real workspace change, the next recovery request '
            'will expose verify for the new revision.'
            if _verification_fix_required(state)
            else (
                'The current recovery request may expose verify because the '
                'workspace is ready for formal validation.'
            )
        )
        return (
            '\n\n[ForgeCode Verification Recovery]\n'
            'The workspace already has task-local changes. The current '
            'completion blocker is missing, stale, insufficient, or failed '
            'formal verification. Do not continue broad repository discovery '
            'or repeat covered reads. If this request only exposes verify, '
            'call it now with the most relevant test, build, lint, or '
            'type-check command for the current project revision. If repair '
            'tools are also exposed, the latest verification failed; use its '
            'output directly to install missing dependencies, edit broken '
            'files, or adjust project scripts. Verification repair permits '
            'only targeted reads/searches within the current Repair Target '
            f'budget before editing or running the concrete repair command. '
            f'{verify_gate}'
            f'{_prefixed_repair_target(repair_target_context)}'
        )
    if _action_recovery_active(state):
        return '\n\n' + render_action_recovery_context(
            _action_recovery_calls(state),
            action_recovery_limit,
            read_used=_action_read_used(state),
        )
    if synthesis_mode == 'checkpoint' or (
        state.runtime is None and state.force_synthesis
    ):
        return (
            '\n\n[ForgeCode Recovery Checkpoint]\n'
            'Recent actions did not produce new evidence or workspace '
            'changes. All listed tools remain available. Reassess the '
            'root goal and existing evidence, then choose a materially '
            'different action. Paths marked as fully covered already have '
            'model-visible or replayable evidence, so do not re-read them '
            'with different line ranges. If your judgment is that the '
            'user goal requires a code change and the Diff is still empty, '
            'use an editing tool once the relevant code is understood. '
            'If exact evidence is missing, perform one targeted search. '
            'If the goal is already satisfied, return a concise final '
            'answer or call finish_task. Do not claim that ForgeCode '
            'paused repository tools.'
        )
    return ''


def _planning_recovery_active(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.control_state is AgentControlState.TASK_PLANNING
    return (
        state.control_state is AgentControlState.TASK_PLANNING
        or (state.control_state is None and state.planning_recovery)
    )


def _planning_recovery_calls(state: RequestState) -> int:
    if state.runtime is not None:
        return state.runtime.planning_recovery_calls
    return state.planning_recovery_calls


def _action_recovery_active(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.control_state is AgentControlState.TARGETED_ANALYSIS
    return (
        state.control_state is AgentControlState.TARGETED_ANALYSIS
        or (state.control_state is None and state.action_recovery)
    )


def _control_state(state: RequestState) -> AgentControlState | None:
    if state.runtime is not None:
        return state.runtime.control_state
    return state.control_state


def _task_contract(state: RequestState) -> TaskContract | None:
    if state.runtime is not None:
        return state.runtime.contract
    return state.task_contract


def _task_goal(state: RequestState) -> str:
    if state.runtime is not None:
        return state.runtime.contract.goal
    return state.task_goal


def _task_scope_patterns(state: RequestState) -> tuple[str, ...]:
    return state.task_scope_patterns


def _change_required(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.contract.requires_change
    return state.change_required


def _mutation_attempted(state: RequestState) -> bool:
    return state.mutation_attempted


def _mutation_recovery_context(state: RequestState) -> str:
    if state.runtime is not None:
        return state.runtime.edit_recovery.context
    return state.mutation_recovery_context


def _mutation_failures(state: RequestState) -> tuple[dict[str, object], ...]:
    if state.runtime is not None:
        return tuple(state.runtime.edit_recovery.failures)
    return state.mutation_failures


def _mutation_read_used(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.edit_recovery.read_used
    return state.mutation_read_used


def _completion_ready_context(state: RequestState) -> str:
    if state.runtime is not None:
        return state.runtime.completion_ready_context
    return state.completion_ready_context


def _synthesis_mode(state: RequestState) -> str:
    if state.runtime is not None:
        return state.runtime.synthesis.mode.value
    if state.finalization_recovery:
        return 'finalization'
    if state.stagnation_final_recovery:
        return 'stagnation_final'
    if state.token_limit_recovery:
        return 'token_limit'
    return ''


def _action_recovery_calls(state: RequestState) -> int:
    if state.runtime is not None:
        return state.runtime.action_recovery_state.calls
    return state.action_recovery_calls


def _action_read_used(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.action_recovery_state.read_used
    return state.action_read_used


def _latest_verification(state: RequestState) -> VerificationEvidence | None:
    if state.runtime is not None:
        return state.runtime.verification.latest
    return state.latest_verification


def _verification_repair_target(state: RequestState) -> RepairTarget | None:
    if state.runtime is not None:
        return state.runtime.verification.repair_target
    return state.verification_repair_target


def _verification_recovery_active(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.verification.recovery_active(
            state.runtime.control_state
        )
    return state.verification_recovery


def _verification_fix_required(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.verification.requires_repair(
            state.runtime.control_state
        )
    return state.verification_fix_required


def _verification_fix_recovery_active(state: RequestState) -> bool:
    if state.runtime is not None:
        return state.runtime.control_state is AgentControlState.FIX_REQUIRED
    return state.verification_fix_recovery


def _verification_read_count(state: RequestState) -> int:
    if state.runtime is not None:
        return state.runtime.verification.read_count
    return state.verification_read_count


def _prefixed_repair_target(context: str) -> str:
    return f'\n\n{context}' if context else ''
