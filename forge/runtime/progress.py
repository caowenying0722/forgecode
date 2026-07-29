'''Deterministic progress evaluation for one agent tool batch.'''

from __future__ import annotations

from dataclasses import dataclass


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
) -> ProgressEvaluation:
    '''Classify whether the last batch materially advanced the task.'''
    if workspace_progressed:
        return ProgressEvaluation(True, 'workspace_revision')
    if verification_progressed:
        return ProgressEvaluation(True, 'verification_evidence')
    if task_progressed:
        return ProgressEvaluation(True, 'task_plan_update')
    if review_progressed:
        return ProgressEvaluation(True, 'diff_review')
    if evidence_progressed:
        return ProgressEvaluation(True, 'repository_evidence')
    if protocol_failure:
        return ProgressEvaluation(False, 'tool_protocol_failure')
    if mutation_recovery_active:
        return ProgressEvaluation(False, 'edit_recovery_active')
    return ProgressEvaluation(False, 'no_new_task_evidence')
