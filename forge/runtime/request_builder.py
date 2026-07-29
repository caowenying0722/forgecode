'''Build one model request from explicit Agent Loop state.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.runtime.intent import TaskContract
from forge.runtime.recovery_feedback import render_action_recovery_context
from forge.runtime.recovery_manager import RecoveryManager
from forge.runtime.task_model import (
    build_runtime_task_model,
    render_runtime_task_model,
)


@dataclass(frozen=True, slots=True)
class RequestState:
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
        if state.tool_free_recovery:
            return None
        if state.planning_recovery:
            return self.recovery_manager.planning_tools()
        if state.finalization_recovery:
            return self.recovery_manager.finalization_tools()
        if (
            state.task_contract is not None
            and state.task_contract.initial_tool_surface == 'read_only'
        ):
            return plan_tools
        if (
            state.task_contract is not None
            and state.task_contract.initial_tool_surface == 'none'
        ):
            return None
        if interaction_mode == 'plan':
            return plan_tools
        if state.verification_recovery:
            return self.recovery_manager.verification_tools(
                fix_available=state.verification_fix_recovery,
                read_available=not state.verification_read_used,
                verify_available=not state.verification_fix_required,
            )
        if state.action_recovery:
            return self.recovery_manager.action_tools(
                read_available=not state.action_read_used
            )
        if state.mutation_failures:
            return self.recovery_manager.mutation_tools(
                list(state.mutation_failures),
                read_available=not state.mutation_read_used,
                include_finish=False,
            )
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
        if state.task_contract is not None and state.task_goal:
            prompt += '\n\n' + render_runtime_task_model(
                build_runtime_task_model(
                    state.task_goal,
                    state.task_contract,
                    scope_patterns=state.task_scope_patterns,
                )
            )
        if state.change_required:
            prompt += '\n\n' + render_change_contract_context(
                changed_paths,
                mutation_attempted=state.mutation_attempted,
                contract=state.task_contract,
                task_scope_patterns=state.task_scope_patterns,
            )
        elif state.task_contract is not None:
            prompt += '\n\n' + render_task_contract_context(
                state.task_contract,
            )
        if state.mutation_recovery_context:
            prompt += '\n\n' + state.mutation_recovery_context
        if state.completion_ready_context:
            prompt += '\n\n' + state.completion_ready_context
        prompt += recovery_system_suffix(
            state,
            action_recovery_limit=self.action_recovery_limit,
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
) -> str:
    if state.finalization_recovery:
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
    if state.planning_recovery:
        return (
            '\n\n[ForgeCode Planning Recovery]\n'
            'The previous tool call was rejected with TODO_REQUIRED. This '
            'request is restricted to todo_write. Call todo_write with a '
            'short working plan and exactly one in_progress item. Do not '
            'request any write, process, verification, discovery, or finish '
            'tool until todo_write succeeds. '
            f'Planning recovery count: {state.planning_recovery_calls}.'
        )
    if state.stagnation_final_recovery:
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
    if state.token_limit_recovery:
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
    if state.action_recovery:
        return '\n\n' + render_action_recovery_context(
            state.action_recovery_calls,
            action_recovery_limit,
            read_used=state.action_read_used,
        )
    if state.verification_recovery:
        verify_gate = (
            'A previous verification failed and no later workspace revision '
            'has been created yet, so verify is intentionally unavailable. '
            'Use the latest verification output to make a relevant repair '
            'first. After a real workspace change, the next recovery request '
            'will expose verify for the new revision.'
            if state.verification_fix_required
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
            'at most one targeted read/search before editing or running the '
            f'concrete repair command. {verify_gate}'
        )
    if state.force_synthesis:
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
