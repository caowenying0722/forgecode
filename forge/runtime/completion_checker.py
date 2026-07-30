'''Agent-level completion decisions composed over CompletionGate.'''

from __future__ import annotations

from typing import Any
import hashlib
import re

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
        self.last_finish_gap_report: dict[str, object] = {}

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
                decision = self._with_relevance_reasons(
                    decision,
                    evidence_paths=evidence_paths,
                )
                reasons.extend(decision.reasons)
            gap_report = self._finish_gap_report(
                verification,
                evidence_paths=evidence_paths,
            )
            self.last_finish_gap_report = gap_report
            missing = (
                tuple(gap_report.get('missing_deliverables', ()))
                + tuple(gap_report.get('missing_acceptance_criteria', ()))
                + tuple(gap_report.get('unfinished_plan_steps', ()))
                + tuple(gap_report.get('missing_runtime_integration', ()))
                + tuple(gap_report.get('missing_verification_types', ()))
            )
            if missing:
                reasons.append(
                    'Completion evidence is insufficient. Gap report: '
                    + repr(gap_report)
                )
        elif mutation_attempted and not changed_paths:
            reasons.append(
                'A workspace write was attempted but produced no final Diff; '
                'continue or declare the task blocked.'
            )
        return tuple(dict.fromkeys(reasons))

    def _finish_gap_report(
        self,
        verification: VerificationEvidence | None,
        *,
        evidence_paths: tuple[str, ...] = (),
    ) -> dict[str, object]:
        task = self.task_manager.active
        if task is None:
            return {}
        changed_paths = (
            self.tracker.changed_paths if self.tracker is not None else ()
        )
        all_paths = tuple(
            dict.fromkeys((*changed_paths, *evidence_paths))
        )
        goal = task.goal
        feature_criteria = _feature_acceptance_criteria(goal)
        text = self._changed_and_evidence_text(all_paths)
        completed_criteria = tuple(
            criterion
            for criterion, pattern in feature_criteria
            if pattern.search(text)
        )
        missing_criteria = tuple(
            criterion
            for criterion, pattern in feature_criteria
            if not pattern.search(text)
        )
        runtime_missing: list[str] = []
        if feature_criteria and not re.search(
            r'(?i)(PlayScene|Scene|create\(|update\(|spawn|runtime|运行时|场景)',
            text,
        ):
            runtime_missing.append('runtime scene/system integration evidence')
        verification_types = _verification_type_evidence(verification)
        missing_verification = []
        if feature_criteria and 'typecheck' not in verification_types:
            missing_verification.append('typecheck')
        if feature_criteria and not (
            {'build', 'test', 'smoke'} & verification_types
        ):
            missing_verification.append('build/test/smoke')
        unfinished_steps = tuple(
            step.title for step in task.steps if step.status != 'completed'
        )
        unrelated = (
            self.tracker.last_classification.unrelated_paths
            if self.tracker is not None
            else ()
        )
        forbidden = (
            self.tracker.last_classification.forbidden_paths
            if self.tracker is not None
            else ()
        )
        return {
            'completed_deliverables': (
                ('task-relevant source/config changes',)
                if changed_paths
                else ()
            ),
            'missing_deliverables': (
                () if changed_paths else ('task-relevant source/config changes',)
            ),
            'satisfied_acceptance_criteria': completed_criteria,
            'missing_acceptance_criteria': missing_criteria,
            'unfinished_plan_steps': unfinished_steps,
            'missing_runtime_integration': tuple(runtime_missing),
            'missing_verification_types': tuple(missing_verification),
            'has_unrelated_or_forbidden_changes': bool(unrelated or forbidden),
            'unrelated_paths': unrelated,
            'forbidden_paths': forbidden,
        }

    def _changed_and_evidence_text(self, paths: tuple[str, ...]) -> str:
        chunks = [' '.join(paths)]
        if self.tracker is None:
            return ' '.join(chunks)
        for path in paths[:20]:
            candidate = self.tracker.root / path
            try:
                if candidate.is_file() and candidate.stat().st_size <= 200_000:
                    chunks.append(candidate.read_text(encoding='utf-8', errors='ignore'))
            except OSError:
                continue
        return '\n'.join(chunks)

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
        if (
            _tracker_filesystem_changed_paths(self.tracker)
            and not self.tracker.changed_paths
        ):
            classification = self.tracker.classifier.classify(
                _tracker_filesystem_changed_paths(self.tracker)
            )
            paths = (
                classification.generated_paths
                + classification.cache_paths
                + classification.unrelated_paths
            )
            if paths:
                return CompletionDecision(
                    allowed=False,
                    reasons=tuple(
                        dict.fromkeys(
                            (
                                *decision.reasons,
                                'The only workspace changes are generated, '
                                'cache, or temporary files and do not satisfy '
                                'the task: '
                                + ', '.join(paths),
                            )
                        )
                    ),
                )
        elif _tracker_filesystem_changed_paths(self.tracker):
            source_paths = set(self.tracker.changed_paths)
            classification = self.tracker.classifier.classify(
                tuple(
                    path
                    for path in _tracker_filesystem_changed_paths(self.tracker)
                    if path not in source_paths
                )
            )
            extra_paths = (
                classification.generated_paths
                + classification.cache_paths
                + classification.unrelated_paths
            )
            if extra_paths:
                decision = CompletionDecision(
                    allowed=False,
                    reasons=tuple(
                        dict.fromkeys(
                            (
                                *decision.reasons,
                                'The workspace also contains generated, '
                                'cache, or temporary files outside the task '
                                'deliverables: '
                                + ', '.join(extra_paths),
                            )
                        )
                    ),
                )
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


def _feature_acceptance_criteria(
    goal: str,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    text = goal.casefold()
    wants_upgrade = bool(re.search(r'三选一|3\s*选\s*1|upgrade|升级', text))
    wants_weapon = bool(re.search(r'武器组合|weapon.*combo|combo|组合', text))
    wants_boss = bool(re.search(r'\bboss\b|Boss|首领|boss', goal))
    criteria: list[tuple[str, re.Pattern[str]]] = []
    if wants_upgrade:
        criteria.extend(
            [
                (
                    'upgrade candidate generation logic exists',
                    re.compile(r'(?i)(upgrade|升级).{0,80}(candidate|option|choice|候选|选项)'),
                ),
                (
                    'runtime presents three upgrade choices',
                    re.compile(r'(?i)(three|3|三).{0,40}(upgrade|choice|option|升级|选项)'),
                ),
                (
                    'player can choose one upgrade',
                    re.compile(r'(?i)(select|choose|pick|选择).{0,80}(upgrade|升级|option|选项)'),
                ),
                (
                    'chosen upgrade mutates player, weapon, or run state',
                    re.compile(r'(?i)(apply|mutate|update|add|修改|应用).{0,100}(player|weapon|state|玩家|武器|状态)'),
                ),
            ]
        )
    if wants_weapon:
        criteria.extend(
            [
                (
                    'weapon combination can be configured and triggered',
                    re.compile(r'(?i)(weapon|武器).{0,80}(combo|combine|synergy|组合|合成|联动)'),
                ),
                (
                    'runtime state can activate at least one weapon combination',
                    re.compile(r'(?i)(trigger|activate|unlock|触发|激活|解锁).{0,100}(combo|combination|组合)'),
                ),
            ]
        )
    if wants_boss:
        criteria.extend(
            [
                (
                    'boss spawn condition exists',
                    re.compile(r'(?i)(boss|首领).{0,100}(spawn|wave|time|score|生成|波次|时间|分数|条件)'),
                ),
                (
                    'boss has independent health, behavior, or phase logic',
                    re.compile(r'(?i)(boss|首领).{0,120}(health|hp|phase|behavior|生命|血量|阶段|行为)'),
                ),
            ]
        )
    if criteria:
        criteria.append(
            (
                'scene or runtime entry point wires these systems together',
                re.compile(r'(?i)(PlayScene|Scene|create\(|update\(|运行时|场景).{0,160}(upgrade|weapon|boss|升级|武器|首领)'),
            )
        )
    return tuple(criteria)


def _verification_type_evidence(
    verification: VerificationEvidence | None,
) -> set[str]:
    if verification is None or not verification.success:
        return set()
    types = {verification.verification_type}
    command = verification.command.casefold()
    if 'tsc' in command and ('--noemit' in command or '--no-emit' in command):
        types.add('typecheck')
    if 'build' in command or 'vite build' in command:
        types.add('build')
    if 'test' in command or 'pytest' in command or 'jest' in command:
        types.add('test')
    return types


def _tracker_filesystem_changed_paths(tracker: WorkspaceTracker) -> tuple[str, ...]:
    return tuple(
        getattr(tracker, 'filesystem_changed_paths', tracker.changed_paths)
    )


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
            source_revision=int(
                metadata.get('source_revision', metadata['workspace_revision'])
            ),
            filesystem_revision=int(metadata.get('filesystem_revision', 0)),
            verification_type=str(metadata.get('verification_type', 'auto')),
            verification_reused=bool(metadata.get('verification_reused', False)),
            generated_artifact_paths=tuple(
                str(path)
                for path in metadata.get('generated_artifact_paths', [])
            ),
            cache_paths=tuple(
                str(path) for path in metadata.get('cache_paths', [])
            ),
            verification_side_effect_paths=tuple(
                str(path)
                for path in metadata.get('verification_side_effect_paths', [])
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
        f'{verification.command} @ source revision '
        f'{verification.bound_source_revision}'
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
        f'{verification.command} @ source revision '
        f'{verification.bound_source_revision}'
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
