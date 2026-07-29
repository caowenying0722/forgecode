'''Agent-level completion decisions composed over CompletionGate.'''

from __future__ import annotations

from typing import Any
import hashlib

from forge.context.working import WorkingState
from forge.runtime.completion import CompletionDecision, CompletionGate
from forge.runtime.state import ToolCall, VerificationEvidence
from forge.runtime.task_scope import (
    evaluate_change_relevance,
    infer_task_scope,
)
from forge.runtime.workspace import WorkspaceTracker
from forge.tasks.manager import TaskManager
from forge.tools.base import ToolResult


VERIFICATION_BLOCKER_MARKERS = (
    'has not been verified with the verify tool',
    'latest verification failed',
    'latest verification timed out',
    'latest verification command was invalid',
    'Project verification is unavailable',
    'changed after verification',
    'verification command does not run',
)


def only_verification_blocked(reasons: tuple[str, ...]) -> bool:
    return bool(reasons) and all(
        any(marker in reason for marker in VERIFICATION_BLOCKER_MARKERS)
        for reason in reasons
    )


class CompletionChecker:
    '''Evaluate finish declarations and stagnation finalization.'''

    def __init__(
        self,
        tracker: WorkspaceTracker | None,
        gate: CompletionGate | None,
        task_manager: TaskManager,
    ) -> None:
        self.tracker = tracker
        self.gate = gate
        self.task_manager = task_manager

    @property
    def available(self) -> bool:
        return self.tracker is not None and self.gate is not None

    @property
    def requires_changes(self) -> bool:
        return self.gate is not None and self.gate.policy.require_changes

    async def evaluate(
        self,
        verification: VerificationEvidence | None,
        *,
        mutation_attempted: bool,
        reviewed_paths: set[str] | None = None,
        evidence_paths: tuple[str, ...] = (),
    ) -> CompletionDecision:
        if self.tracker is None or self.gate is None:
            return CompletionDecision(allowed=True)
        decision = await self.gate.evaluate(
            self.tracker,
            verification,
            mutation_attempted=mutation_attempted,
            reviewed_paths=reviewed_paths,
        )
        if not decision.allowed:
            return decision
        return self._with_relevance_reasons(
            decision,
            evidence_paths=evidence_paths,
        )

    async def finish_rejection_reasons(
        self,
        result: ToolResult,
        *,
        working_state: WorkingState,
        mutation_attempted: bool,
        change_required: bool,
        verification: VerificationEvidence | None,
        reviewed_paths: set[str] | None = None,
        evidence_paths: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        metadata = result.metadata
        if metadata.get('status') == 'blocked':
            if working_state.has_external_blocker:
                return ()
            return (
                'blocked is reserved for an external condition that requires '
                'user action, permission, credentials, or an unavailable '
                'dependency. Repeated reads, malformed arguments, lack of '
                'progress, and ForgeCode recovery guidance are not blockers; '
                'continue with the available tools.',
            )
        task_kind = str(metadata.get('task_kind', ''))
        reasons: list[str] = []
        changed_paths = self.tracker.changed_paths if self.tracker else ()
        if change_required and task_kind != 'change' and not changed_paths:
            reasons.append(
                'This turn requires a real task-local workspace change. '
                'Inspection or answer completion cannot satisfy it while '
                'the task-local Diff is empty.'
            )
        if task_kind == 'inspection' and not working_state.evidence_paths:
            reasons.append(
                'An inspection task requires repository evidence from '
                'read_file, list_directory, grep, or find_files.'
            )
        if task_kind != 'change' and changed_paths:
            reasons.append(
                'The workspace changed during this turn; declare '
                'task_kind=change and provide current verification evidence.'
            )
        if task_kind == 'change':
            if self.tracker is None or self.gate is None:
                reasons.append(
                    'Workspace tracking is unavailable, so a change outcome '
                    'cannot be verified.'
                )
            else:
                decision = await self.gate.evaluate(
                    self.tracker,
                    verification,
                    mutation_attempted=True,
                    reviewed_paths=reviewed_paths,
                )
                reasons.extend(decision.reasons)
                if decision.allowed:
                    reasons.extend(
                        self._with_relevance_reasons(
                            decision,
                            evidence_paths=evidence_paths,
                        ).reasons
                    )
        elif mutation_attempted and not changed_paths:
            reasons.append(
                'A workspace write was attempted but produced no final Diff; '
                'continue or declare the task blocked.'
            )
        return tuple(dict.fromkeys(reasons))

    async def can_finalize_after_stagnation(
        self,
        *,
        mutation_attempted: bool,
        verification: VerificationEvidence | None,
        mutation_failures: list[dict[str, object]],
        reviewed_paths: set[str] | None = None,
        evidence_paths: tuple[str, ...] = (),
    ) -> bool:
        if (
            self.tracker is None
            or self.gate is None
            or not self.tracker.changed_paths
            or mutation_failures
        ):
            return False
        task = self.task_manager.active
        if task is not None and task.planned and any(
            step.status != 'completed' for step in task.steps
        ):
            return False
        decision = await self.gate.evaluate(
            self.tracker,
            verification,
            mutation_attempted=mutation_attempted,
            reviewed_paths=reviewed_paths,
        )
        if not decision.allowed:
            return False
        return self._with_relevance_reasons(
            decision,
            evidence_paths=evidence_paths,
        ).allowed

    def task_scope_patterns(
        self,
        *,
        evidence_paths: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        task = self.task_manager.active
        if task is None:
            return ()
        return infer_task_scope(
            task.goal,
            evidence_paths=evidence_paths,
            scope_hints=task.scope_hints,
        ).patterns

    def _with_relevance_reasons(
        self,
        decision: CompletionDecision,
        *,
        evidence_paths: tuple[str, ...] = (),
    ) -> CompletionDecision:
        if self.tracker is None:
            return decision
        task = self.task_manager.active
        if task is None:
            return decision
        scope = infer_task_scope(
            task.goal,
            evidence_paths=evidence_paths,
            scope_hints=task.scope_hints,
        )
        relevance = evaluate_change_relevance(
            self.tracker.changed_paths,
            scope,
        )
        if relevance.relevant:
            return decision
        return CompletionDecision(
            allowed=False,
            reasons=tuple(
                dict.fromkeys((*decision.reasons, *relevance.reasons))
            ),
        )


def build_completion_feedback(
    reasons: tuple[str, ...],
    *,
    task_context: str = '',
) -> dict[str, Any]:
    details = '\n'.join(f'- {reason}' for reason in reasons)
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n'
            'ForgeCode completion check rejected the previous final answer.\n'
            f'{details}\n'
            'Follow the verification state machine. If verification is '
            'missing or stale, call verify now using target=auto or a '
            'discovered command_id. If the latest verification failed, use '
            'the failure output to repair the relevant code or configuration, '
            'then call verify again. If the verification command was invalid, '
            'do not repeat it; use a discovered non-interactive project '
            'validation command.'
        ),
    }


def verification_from_result(
    result: ToolResult,
) -> VerificationEvidence | None:
    metadata = result.metadata
    if metadata.get('verification') is not True:
        return None
    status = str(metadata.get('verification_status', ''))
    if not status:
        timed_out = bool(metadata.get('timed_out', False))
        exit_code = int(metadata.get('exit_code', 0))
        status = 'timed_out' if timed_out else (
            'passed' if exit_code == 0 else 'failed'
        )
    signature_text = '\n'.join(
        str(metadata.get(key, '')) for key in ('command', 'exit_code', 'stderr')
    )
    try:
        return VerificationEvidence(
            command=str(metadata['command']),
            cwd=str(metadata['cwd']),
            exit_code=int(metadata['exit_code']),
            duration_seconds=float(metadata['duration_seconds']),
            timed_out=bool(metadata['timed_out']),
            workspace_revision=int(metadata['workspace_revision']),
            status=status,
            command_id=str(metadata.get('command_id', '')),
            failure_signature=(
                hashlib.sha256(signature_text.encode('utf-8')).hexdigest()
                if status != 'passed'
                else ''
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def completion_review_paths(
    tool_results: list[tuple[ToolCall, ToolResult]],
    changed_paths: tuple[str, ...],
) -> set[str]:
    changed = {path.replace('\\', '/') for path in changed_paths}
    reviewed: set[str] = set()
    for tool_call, result in tool_results:
        if (
            tool_call.name != 'git_diff'
            or not result.success
            or not result.content.strip()
        ):
            continue
        path = result.metadata.get('path')
        if path is None:
            reviewed.update(changed)
            continue
        normalized = str(path).replace('\\', '/')
        if normalized in changed:
            reviewed.add(normalized)
    return reviewed


def render_completion_ready_context(
    changed_paths: tuple[str, ...],
    verification: VerificationEvidence | None,
    decision_calls: int,
    decision_limit: int,
    reviewed_paths: set[str],
) -> str:
    changed = ', '.join(changed_paths)
    reviewed = ', '.join(sorted(reviewed_paths)) or 'none'
    verification_status = (
        f'{verification.command} @ revision {verification.workspace_revision}'
        if verification is not None
        else 'not required / not run'
    )
    remaining = max(decision_limit - decision_calls, 0)
    return (
        '[ForgeCode Completion Ready]\n'
        f'changed paths: {changed}\n'
        f'current verification: {verification_status}\n'
        f'reviewed Diff paths: {reviewed}\n'
        f'decision calls remaining: {remaining}\n'
        'Deterministic completion checks pass for the current revision. '
        'All tools listed in this request remain available, but open-ended '
        'discovery is no longer useful. Decide whether the user goal is '
        'satisfied. If it is, return the final answer or call finish_task '
        'alone. If it is not, make one concrete workspace edit based on the '
        'existing evidence, then verify the new revision. You may call one '
        'scoped git_diff only for a changed path not already reviewed.'
    )


def build_finalization_recovery_feedback(
    task_context: str,
    working_context: str,
    changed_paths: tuple[str, ...],
    verification: VerificationEvidence | None,
) -> dict[str, Any]:
    verification_status = (
        f'{verification.command} @ revision {verification.workspace_revision}'
        if verification is not None
        else 'not required / not run'
    )
    changed = ', '.join(changed_paths)
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n{working_context}\n\n'
            '[ForgeCode Finalization Recovery]\n'
            'The current revision passed every deterministic completion '
            'check, but the trajectory continued diagnostics without another '
            'workspace change. The next request is a dedicated final '
            'synthesis with no tools. Return a concise final answer in the '
            "user's language. Summarize the actual changed paths "
            f'({changed}) and verification ({verification_status}). State any '
            'semantic or visual limitation honestly. Do not request another '
            'tool call.'
        ),
    }
