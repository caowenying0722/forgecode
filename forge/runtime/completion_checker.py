'''Agent-level completion decisions composed over CompletionGate.'''

from __future__ import annotations

from typing import Any
import hashlib
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
import re

from forge.context.working import WorkingState
from forge.runtime.acceptance import AcceptanceLedger
from forge.runtime.completion import (
    CompletionDecision,
    CompletionGate,
    matches_any,
)
from forge.runtime.intent import TaskContract
from forge.runtime.state import (
    ToolCall,
    VerificationEvidence,
    normalize_verification_levels,
)
from forge.runtime.task_scope import (
    TaskScope,
    evaluate_change_relevance,
    infer_task_scope,
)
from forge.runtime.workspace import WorkspaceTracker, fingerprint_path
from forge.runtime.workspace_classification import artifact_deltas_from_metadata
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
        *,
        acceptance_ledger: AcceptanceLedger | None = None,
    ) -> None:
        self.tracker = tracker
        self.gate = gate
        self.task_manager = task_manager
        self.acceptance_ledger = acceptance_ledger or AcceptanceLedger()
        self.task_contract: TaskContract | None = None
        self.last_finish_gap_report: dict[str, object] = {}

    @property
    def available(self) -> bool:
        return self.tracker is not None and self.gate is not None

    @property
    def requires_changes(self) -> bool:
        # Deprecated compatibility surface. Global policy must not force the
        # current prompt into a change contract; CompletionGate enforces policy
        # after the current turn is independently classified.
        return False

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
            verification=verification,
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
        declaration_status = str(metadata.get('status', ''))
        remaining_work = tuple(
            str(item).strip()
            for item in metadata.get('remaining_work', [])
            if str(item).strip()
        )
        if declaration_status in {'completed', 'task_completed'} and remaining_work:
            reasons.append(
                'The structured finish declaration still has remaining work: '
                + ', '.join(remaining_work)
            )
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
        if (
            task_kind == 'inspection'
            and working_state.evidence_paths
            and not working_state.answer_mentions_evidence(
                str(metadata.get('summary', ''))
            )
        ):
            reasons.append(
                'An inspection completion must reference collected repository '
                'evidence. Mention at least one of: '
                + ', '.join(working_state.evidence_paths[:10])
            )
        if task_kind != 'change' and changed_paths:
            reasons.append(
                'The workspace changed during this turn; declare '
                'task_kind=change and provide current verification evidence.'
            )
        if declaration_status in {
            'progressed',
            'step_completed',
            'partially_completed',
        }:
            if not remaining_work:
                reasons.append(
                    'A non-terminal finish outcome requires structured '
                    'remaining_work.'
                )
            if task_kind == 'change' and mutation_attempted and not changed_paths:
                reasons.append(
                    'A workspace write was attempted but produced no final Diff.'
                )
            return tuple(dict.fromkeys(reasons))
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
                    verification=verification,
                    evidence_paths=evidence_paths,
                )
                reasons.extend(decision.reasons)
            gap_report = self._finish_gap_report(
                verification,
                evidence_paths=evidence_paths,
            )
            self.last_finish_gap_report = gap_report
            missing = (
                tuple(gap_report.get('pending_deliverables', ()))
                + tuple(gap_report.get('missing_criteria', ()))
                + tuple(gap_report.get('pending_plan_steps', ()))
                + tuple(gap_report.get('pending_plan_deliverables', ()))
                + tuple(gap_report.get('missing_verification', ()))
                + tuple(gap_report.get('unresolved_repair_target', ()))
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
        contract = self.task_contract
        ledger_report = self.acceptance_ledger.gap_report()
        pending_steps = tuple(
            step.title for step in task.steps if step.status != 'completed'
        )
        pending_plan_deliverables = _pending_plan_deliverables(
            task.steps,
            changed_paths=changed_paths,
            evidence_paths=evidence_paths,
        )
        current_classification = (
            self.tracker.classifier.classify(
                _tracker_filesystem_changed_paths(self.tracker)
            )
            if self.tracker is not None
            else None
        )
        unrelated = (
            current_classification.unrelated_paths
            if current_classification is not None
            else ()
        )
        forbidden = (
            current_classification.forbidden_paths
            if current_classification is not None
            else ()
        )
        deliverables = (
            contract.deliverables
            if contract is not None
            else ('workspace changes',)
        )
        satisfied_criteria = tuple(ledger_report['satisfied_criteria'])
        partial_criteria = tuple(
            ledger_report['partially_satisfied_criteria']
        )
        missing_criteria = tuple(ledger_report['missing_criteria'])
        completed_deliverables, pending_deliverables = _deliverable_status(
            deliverables,
            changed_paths=changed_paths,
            evidence_paths=evidence_paths,
            missing_criteria=missing_criteria,
            pending_steps=pending_steps,
        )
        missing_verification = _missing_verification(
            contract,
            verification,
        )
        unresolved_repair_target: tuple[str, ...] = ()
        recommended = _recommended_next_action(
            pending_deliverables=pending_deliverables,
            missing_criteria=missing_criteria,
            missing_verification=missing_verification,
            pending_steps=pending_steps,
            unrelated=unrelated,
            forbidden=forbidden,
        )
        return {
            'completed_deliverables': completed_deliverables,
            'pending_deliverables': pending_deliverables,
            'satisfied_criteria': satisfied_criteria,
            'partially_satisfied_criteria': partial_criteria,
            'missing_criteria': missing_criteria,
            'missing_evidence': tuple(ledger_report['missing_evidence']),
            'pending_plan_steps': pending_steps,
            'pending_plan_deliverables': pending_plan_deliverables,
            'missing_verification': missing_verification,
            'unresolved_repair_target': unresolved_repair_target,
            'unrelated_changes': unrelated,
            'forbidden_changes': forbidden,
            'recommended_next_action': recommended,
            # Compatibility aliases for older callers/tests.
            'missing_deliverables': pending_deliverables,
            'satisfied_acceptance_criteria': satisfied_criteria,
            'missing_acceptance_criteria': missing_criteria,
            'unfinished_plan_steps': pending_steps,
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
        if (
            self.acceptance_ledger.partial_ids()
            or self.acceptance_ledger.pending_ids()
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
            verification=verification,
            evidence_paths=evidence_paths,
        ).allowed

    def task_scope_patterns(
        self,
        *,
        evidence_paths: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return self.task_scope(evidence_paths=evidence_paths).patterns

    def task_scope(
        self,
        *,
        evidence_paths: tuple[str, ...] = (),
    ) -> TaskScope:
        task = self.task_manager.active
        if task is None:
            return TaskScope()
        return infer_task_scope(
            task.goal,
            evidence_paths=evidence_paths,
            scope_hints=task.scope_hints,
        )

    def _with_relevance_reasons(
        self,
        decision: CompletionDecision,
        *,
        verification: VerificationEvidence | None = None,
        evidence_paths: tuple[str, ...] = (),
    ) -> CompletionDecision:
        if self.tracker is None:
            return decision
        task = self.task_manager.active
        if task is None:
            return decision
        filesystem_paths = _tracker_filesystem_changed_paths(self.tracker)
        trusted_outputs = trusted_verification_output_paths(
            self.tracker,
            verification,
            forbidden_patterns=(
                self.gate.policy.forbidden_paths if self.gate is not None else ()
            ),
        )
        integrity_violations = verification_artifact_integrity_violations(
            self.tracker,
            verification,
            forbidden_patterns=(
                self.gate.policy.forbidden_paths if self.gate is not None else ()
            ),
        )
        if integrity_violations:
            decision = CompletionDecision(
                allowed=False,
                reasons=tuple(
                    dict.fromkeys(
                        (
                            *decision.reasons,
                            'Verified artifact state changed after '
                            'verification: '
                            + ', '.join(integrity_violations),
                        )
                    )
                ),
            )
        untrusted_filesystem_paths = tuple(
            path for path in filesystem_paths if path not in trusted_outputs
        )
        if filesystem_paths and not self.tracker.changed_paths:
            classification = self.tracker.classifier.classify(
                untrusted_filesystem_paths
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
        elif filesystem_paths:
            source_paths = set(self.tracker.changed_paths)
            classification = self.tracker.classifier.classify(
                tuple(
                    path
                    for path in untrusted_filesystem_paths
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


def _deliverable_status(
    deliverables: tuple[str, ...],
    *,
    changed_paths: tuple[str, ...],
    evidence_paths: tuple[str, ...],
    missing_criteria: tuple[str, ...],
    pending_steps: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    completed: list[str] = []
    pending: list[str] = []
    for deliverable in deliverables:
        normalized = deliverable.casefold()
        if any(
            token in normalized
            for token in ('workspace', 'change', 'diff', 'code', 'source')
        ):
            (completed if changed_paths else pending).append(deliverable)
        elif 'analysis' in normalized or 'answer' in normalized:
            (completed if evidence_paths or not missing_criteria else pending).append(
                deliverable
            )
        elif not missing_criteria and not pending_steps:
            completed.append(deliverable)
        else:
            pending.append(deliverable)
    return tuple(completed), tuple(pending)


def _pending_plan_deliverables(
    steps: tuple[object, ...],
    *,
    changed_paths: tuple[str, ...],
    evidence_paths: tuple[str, ...],
) -> tuple[str, ...]:
    available_paths = {
        path.replace('\\', '/') for path in (*changed_paths, *evidence_paths)
    }
    pending: list[str] = []
    for step in steps:
        if getattr(step, 'status', '') != 'completed':
            continue
        title = str(getattr(step, 'title', 'step'))
        step_evidence = tuple(
            str(item).replace('\\', '/')
            for item in getattr(step, 'evidence', ())
        )
        for deliverable in getattr(step, 'deliverables', ()):
            value = str(deliverable).strip()
            if not value:
                continue
            if not _deliverable_matches_evidence(
                value,
                paths=available_paths,
                step_evidence=step_evidence,
            ):
                pending.append(f'{title}: {value}')
    return tuple(dict.fromkeys(pending))


def _deliverable_matches_evidence(
    deliverable: str,
    *,
    paths: set[str],
    step_evidence: tuple[str, ...],
) -> bool:
    normalized = deliverable.replace('\\', '/').strip()
    if (
        '/' in normalized
        or '*' in normalized
        or PurePosixPath(normalized).suffix
    ):
        candidates = (*paths, *step_evidence)
        return any(
            candidate == normalized
            or fnmatchcase(candidate, normalized)
            or fnmatchcase(normalized, candidate)
            for candidate in candidates
        )
    terms = tuple(
        token
        for token in re.findall(r'[A-Za-z0-9_\u4e00-\u9fff]+', normalized.casefold())
        if len(token) >= 3
    )
    evidence_text = ' '.join(step_evidence).casefold()
    return bool(terms and all(term in evidence_text for term in terms))


def _missing_verification(
    contract: TaskContract | None,
    verification: VerificationEvidence | None,
) -> tuple[str, ...]:
    if contract is None or not contract.verification_policy.required:
        return ()
    if verification is None:
        return ('current verification evidence',)
    if not verification.success:
        return ('passing verification evidence',)
    return ()


def _recommended_next_action(
    *,
    pending_deliverables: tuple[str, ...],
    missing_criteria: tuple[str, ...],
    missing_verification: tuple[str, ...],
    pending_steps: tuple[str, ...],
    unrelated: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> str:
    if forbidden:
        return 'Revert or isolate forbidden changes before finishing.'
    if unrelated:
        return 'Remove unrelated changes or connect them to the active task scope.'
    if pending_steps:
        return 'Complete the next pending plan step with structured evidence.'
    if missing_criteria:
        return 'Produce evidence for the next missing acceptance criterion.'
    if missing_verification:
        return 'Run current verification for the final source revision.'
    if pending_deliverables:
        return 'Complete the pending deliverable with task-local evidence.'
    return 'Declare completion with a concise summary.'


def _verification_type_evidence(
    verification: VerificationEvidence | None,
) -> set[str]:
    if verification is None or not verification.success:
        return set()
    types = {verification.verification_type}
    level_types = {
        'typecheck_verified': 'typecheck',
        'unit_tests_verified': 'test',
        'build_verified': 'build',
        'dev_server_verified': 'smoke',
        'browser_smoke_verified': 'smoke',
        'interaction_verified': 'runtime_integration',
        'acceptance_verified': 'acceptance',
    }
    types.update(
        level_types[level]
        for level in verification.verification_levels
        if level in level_types
    )
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
            generated_artifact_fingerprints=_fingerprints_from_metadata(
                metadata.get('generated_artifact_fingerprints', [])
            ),
            cache_fingerprints=_fingerprints_from_metadata(
                metadata.get('cache_fingerprints', [])
            ),
            artifact_deltas=artifact_deltas_from_metadata(
                metadata.get('artifact_deltas', [])
            ),
            verification_side_effect_paths=tuple(
                str(path)
                for path in metadata.get('verification_side_effect_paths', [])
            ),
            verification_levels=normalize_verification_levels(
                metadata.get('verification_levels', [])
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def trusted_verification_output_paths(
    tracker: WorkspaceTracker,
    verification: VerificationEvidence | None,
    *,
    forbidden_patterns: tuple[str, ...] = (),
) -> frozenset[str]:
    if verification is None or not verification.success:
        return frozenset()
    if verification.bound_source_revision != tracker.source_revision:
        return frozenset()
    if verification.verification_side_effect_paths:
        return frozenset()
    filesystem_paths = set(_tracker_filesystem_changed_paths(tracker))
    fingerprint_by_path = {
        path: fingerprint
        for path, fingerprint in (
            *verification.generated_artifact_fingerprints,
            *verification.cache_fingerprints,
        )
    }
    declared = set(verification.generated_artifact_paths) | set(
        verification.cache_paths
    )
    trusted: set[str] = set()
    for delta in verification.artifact_deltas:
        path = delta.path.replace('\\', '/')
        if not _is_workspace_relative_path(path):
            continue
        if forbidden_patterns and matches_any(path, forbidden_patterns):
            continue
        if not delta.rule_pattern or not matches_any(
            path,
            (delta.rule_pattern,),
        ):
            continue
        current_fingerprint = fingerprint_path(tracker.root, path)
        if delta.operation == 'deleted':
            if current_fingerprint != 'missing':
                continue
        elif (
            current_fingerprint != 'missing'
            and current_fingerprint != delta.after_fingerprint
        ):
            continue
        trusted.add(path)
    for raw_path in declared:
        path = str(raw_path).replace('\\', '/')
        if path not in filesystem_paths:
            continue
        if not _is_workspace_relative_path(path):
            continue
        if forbidden_patterns and matches_any(path, forbidden_patterns):
            continue
        fingerprint = fingerprint_by_path.get(path)
        if not fingerprint:
            continue
        if tracker.current.files.get(path) != fingerprint:
            continue
        trusted.add(path)
    return frozenset(trusted)


def verification_artifact_integrity_violations(
    tracker: WorkspaceTracker,
    verification: VerificationEvidence | None,
    *,
    forbidden_patterns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    '''Return verified outputs whose current state contradicts their delta.'''
    if verification is None or not verification.success:
        return ()
    if verification.bound_source_revision != tracker.source_revision:
        return ()
    if verification.verification_side_effect_paths:
        return ()
    violations: list[str] = []
    for delta in verification.artifact_deltas:
        path = delta.path.replace('\\', '/')
        if not _is_workspace_relative_path(path):
            violations.append(path)
            continue
        if forbidden_patterns and matches_any(path, forbidden_patterns):
            violations.append(path)
            continue
        if not delta.rule_pattern or not matches_any(
            path,
            (delta.rule_pattern,),
        ):
            violations.append(path)
            continue
        current = fingerprint_path(tracker.root, path)
        if delta.operation == 'deleted':
            if current != 'missing':
                violations.append(path)
        elif current not in {'missing', delta.after_fingerprint}:
            violations.append(path)
    return tuple(dict.fromkeys(violations))


def _is_workspace_relative_path(path: str) -> bool:
    if not path or path.startswith('/') or '\\' in path:
        return False
    parts = PurePosixPath(path).parts
    return '..' not in parts and not any(':' in part for part in parts)


def _fingerprints_from_metadata(value: object) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for item in value:
        if isinstance(item, dict):
            path = item.get('path')
            fingerprint = item.get('fingerprint')
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            path, fingerprint = item
        else:
            continue
        if path is None or fingerprint is None:
            continue
        pairs.append((str(path), str(fingerprint)))
    return tuple(pairs)


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
