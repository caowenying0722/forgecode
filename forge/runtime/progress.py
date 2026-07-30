'''Deterministic progress evaluation for one agent tool batch.'''

from __future__ import annotations

from dataclasses import dataclass

from forge.runtime.completion import matches_any
from forge.runtime.task_scope import DISPOSABLE_PATH_PATTERNS
from forge.runtime.workspace_classification import WorkspaceChangeClassifier


@dataclass(frozen=True, slots=True)
class ProgressEvaluation:
    progressed: bool
    signal: str


def evaluate_progress(
    *,
    workspace_progressed: bool,
    task_progressed: bool,
    evidence_progressed: bool,
    verification_progressed: bool,
    review_progressed: bool,
    protocol_failure: bool,
    mutation_recovery_active: bool,
    requires_change: bool = False,
    task_scope_patterns: tuple[str, ...] = (),
    changed_paths: tuple[str, ...] = (),
    evidence_paths: tuple[str, ...] = (),
    review_paths: tuple[str, ...] = (),
    repair_target_paths: tuple[str, ...] = (),
    source_changed_paths: tuple[str, ...] = (),
    completed_acceptance_criteria: tuple[str, ...] = (),
    completed_plan_step: bool = False,
    previous_verification_error_count: int | None = None,
    current_verification_error_count: int | None = None,
    failure_signature_changed: bool = False,
    verification_reused: bool = False,
    repair_target_resolved: bool = False,
) -> ProgressEvaluation:
    '''Classify whether the last batch materially advanced the task.'''
    if completed_acceptance_criteria:
        return ProgressEvaluation(True, 'acceptance_criterion')
    if completed_plan_step:
        return ProgressEvaluation(True, 'plan_step')
    if repair_target_resolved:
        return ProgressEvaluation(True, 'repair_target_resolved')
    if workspace_progressed:
        effective_source_paths = source_changed_paths or changed_paths
        if not effective_source_paths:
            return ProgressEvaluation(False, 'filesystem_revision_only')
        if not _paths_relevant(
            effective_source_paths,
            task_scope_patterns=task_scope_patterns,
            repair_target_paths=repair_target_paths,
        ):
            return ProgressEvaluation(False, 'unrelated_workspace_revision')
        return ProgressEvaluation(True, 'workspace_revision')
    if verification_progressed:
        if (
            previous_verification_error_count is not None
            and current_verification_error_count is not None
            and current_verification_error_count < previous_verification_error_count
        ):
            return ProgressEvaluation(True, 'verification_error_reduced')
        if failure_signature_changed:
            return ProgressEvaluation(True, 'verification_signature_changed')
        if repair_target_paths and not _paths_relevant(
            changed_paths,
            task_scope_patterns=task_scope_patterns,
            repair_target_paths=repair_target_paths,
        ):
            return ProgressEvaluation(False, 'unrelated_verification_evidence')
        if verification_reused:
            return ProgressEvaluation(False, 'reused_verification_evidence')
        return ProgressEvaluation(True, 'verification_evidence')
    if task_progressed:
        return ProgressEvaluation(True, 'task_plan_update')
    if review_progressed:
        if not _paths_relevant(
            review_paths or changed_paths,
            task_scope_patterns=task_scope_patterns,
            repair_target_paths=repair_target_paths,
        ):
            return ProgressEvaluation(False, 'unrelated_diff_review')
        return ProgressEvaluation(True, 'diff_review')
    if evidence_progressed:
        if requires_change and not _paths_relevant(
            evidence_paths,
            task_scope_patterns=task_scope_patterns,
            repair_target_paths=repair_target_paths,
        ):
            return ProgressEvaluation(False, 'unrelated_repository_evidence')
        return ProgressEvaluation(True, 'repository_evidence')
    if protocol_failure:
        return ProgressEvaluation(False, 'tool_protocol_failure')
    if mutation_recovery_active:
        return ProgressEvaluation(False, 'edit_recovery_active')
    return ProgressEvaluation(False, 'no_new_task_evidence')


def _paths_relevant(
    paths: tuple[str, ...],
    *,
    task_scope_patterns: tuple[str, ...],
    repair_target_paths: tuple[str, ...],
) -> bool:
    normalized = tuple(path.replace('\\', '/') for path in paths if path)
    if not normalized:
        return not (task_scope_patterns or repair_target_paths)
    classification = WorkspaceChangeClassifier().classify(normalized)
    if not classification.task_candidate_paths:
        return False
    if all(matches_any(path, DISPOSABLE_PATH_PATTERNS) for path in normalized):
        return False
    if any(matches_any(path, DISPOSABLE_PATH_PATTERNS) for path in normalized):
        return False
    allowed = (*repair_target_paths, *task_scope_patterns)
    if not allowed:
        return True
    return all(_matches_scope(path, allowed) for path in normalized)

def _matches_scope(path: str, patterns: tuple[str, ...]) -> bool:
    if matches_any(path, patterns):
        return True
    return any(
        pattern.endswith('/**') and path == pattern[:-3].rstrip('/')
        for pattern in patterns
    )
