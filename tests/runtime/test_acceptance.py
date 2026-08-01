'''Generic acceptance evidence and progress regression tests.'''

from __future__ import annotations

import asyncio
from pathlib import Path

from forge.runtime.acceptance import (
    AcceptanceEvidence,
    AcceptanceLedger,
)
from forge.runtime.completion_checker import CompletionChecker
from forge.runtime.intent import (
    TaskContract,
    TurnIntent,
    VerificationPolicy,
)
from forge.runtime.progress import evaluate_progress
from forge.runtime.state import AcceptanceLedgerUpdated, VerificationEvidence
from forge.sessions.store import SessionStore
from forge.tasks.manager import TaskManager
from forge.tools.base import ToolRegistry, ToolResult
from forge.tools.task import create_task_tools


def change_contract(
    *,
    criteria: tuple[str, ...],
    deliverables: tuple[str, ...] = ('workspace changes',),
) -> TaskContract:
    return TaskContract(
        intent=TurnIntent('implement', 'high', 'test contract'),
        requires_change=True,
        requires_plan=False,
        completion_contract='change',
        initial_phase='implementing',
        initial_tool_surface='all',
        deliverables=deliverables,
        acceptance_criteria=criteria,
        verification_policy=VerificationPolicy(kind='required', required=True),
    )


def test_task_contract_acceptance_criteria_are_tracked() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=(
                'A source diff exists.',
                'Typecheck passes.',
            )
        )
    )

    completed = ledger.observe_source_change(
        ('src/app.ts',),
        source_revision=1,
    )
    assert completed == ('criterion-1',)
    assert ledger.criteria['criterion-1'].status == 'satisfied'
    assert ledger.criteria['criterion-2'].status == 'partially_satisfied'

    verified = ledger.observe_verification(
        VerificationEvidence(
            command='npx tsc --noEmit',
            cwd='.',
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            workspace_revision=1,
            source_revision=1,
            verification_type='typecheck',
        )
    )
    assert verified == ('criterion-2',)
    assert ledger.satisfied_ids() == ('criterion-1', 'criterion-2')


def test_game_terms_in_source_do_not_satisfy_behavior_criterion(
    tmp_path: Path,
) -> None:
    del tmp_path
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=(
                'Runtime behavior has smoke evidence.',
            )
        )
    )

    completed = ledger.observe_source_change(
        ('src/game.ts',),
        source_revision=1,
        symbols=('upgradeBossStringOnly',),
    )

    assert completed == ()
    assert ledger.criteria['criterion-1'].status == 'partially_satisfied'


def test_task_update_without_structured_evidence_is_not_progress(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Implement generic behavior')
    ledger = AcceptanceLedger.from_contract(
        change_contract(criteria=('Pytest passes.',))
    )
    registry = ToolRegistry(
        create_task_tools(
            tmp_path,
            manager,
            acceptance_ledger=ledger,
        )
    )
    asyncio.run(
        registry.execute(
            'task_plan',
            {
                'steps': [
                    {
                        'title': 'Implement behavior',
                        'criterion_ids': ['criterion-1'],
                    },
                    'Verify behavior',
                ]
            },
        )
    )

    result = asyncio.run(
        registry.execute(
            'task_update',
            {
                'step_id': 'step-1',
                'status': 'completed',
                'evidence': ['Done.'],
            },
        )
    )
    progress = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=False,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
    )

    assert result.success is True
    assert result.metadata['evidence_valid'] is False
    assert result.metadata['completed_plan_step'] is False
    assert progress.progressed is False


def test_task_update_with_source_and_verification_evidence_progresses(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Fix Python bug')
    ledger = AcceptanceLedger.from_contract(
        change_contract(criteria=('Pytest passes.',))
    )
    registry = ToolRegistry(
        create_task_tools(
            tmp_path,
            manager,
            acceptance_ledger=ledger,
        )
    )
    asyncio.run(
        registry.execute(
            'task_plan',
            {
                'steps': [
                    {
                        'title': 'Fix division behavior',
                        'criterion_ids': ['criterion-1'],
                    },
                    'Verify fix',
                ]
            },
        )
    )

    result = asyncio.run(
        registry.execute(
            'task_update',
            {
                'step_id': 'step-1',
                'status': 'completed',
                'acceptance_evidence': [
                    {
                        'criterion_id': 'criterion-1',
                        'evidence_type': 'test_result',
                        'evidence_paths': ['tests/test_math.py'],
                        'verification_record_ids': ['pytest:test:1'],
                        'source_revision': 1,
                        'producer': 'test',
                        'confidence': 0.95,
                        'explanation': 'Pytest covered the fixed behavior.',
                    }
                ],
            },
        )
    )
    progress = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=False,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
        completed_plan_step=bool(result.metadata['completed_plan_step']),
        completed_acceptance_criteria=tuple(
            result.metadata['completed_criterion_ids']
        ),
    )

    assert result.metadata['evidence_valid'] is True
    assert result.metadata['completed_plan_step'] is True
    assert result.metadata['completed_criterion_ids'] == ['criterion-1']
    assert progress.progressed is True
    assert progress.signal == 'acceptance_criterion'


def test_verification_error_reduction_and_reused_failure_progress() -> None:
    reduced = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=True,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
        previous_verification_error_count=3,
        current_verification_error_count=1,
    )
    repeated = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=True,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
        previous_verification_error_count=1,
        current_verification_error_count=1,
        verification_reused=True,
    )

    assert reduced.progressed is True
    assert reduced.signal == 'verification_error_reduced'
    assert repeated.progressed is False
    assert repeated.signal == 'reused_verification_evidence'


def test_repair_target_resolution_counts_as_progress() -> None:
    progress = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=False,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
        repair_target_resolved=True,
    )

    assert progress.progressed is True
    assert progress.signal == 'repair_target_resolved'


def test_build_passes_only_build_criterion() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=(
                'Build passes.',
                'Runtime behavior has smoke evidence.',
            )
        )
    )
    completed = ledger.observe_verification(
        VerificationEvidence(
            command='npm run build',
            cwd='.',
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            workspace_revision=1,
            source_revision=1,
            verification_type='build',
        )
    )

    assert completed == ('criterion-1',)
    assert ledger.criteria['criterion-1'].status == 'satisfied'
    assert ledger.criteria['criterion-2'].status == 'pending'


def test_business_behavior_without_smoke_stays_partial() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=('Runtime behavior has smoke evidence.',)
        )
    )

    ledger.observe_source_change(('src/runtime.ts',), source_revision=1)

    assert ledger.criteria['criterion-1'].status == 'partially_satisfied'


def test_build_success_alone_does_not_satisfy_gameplay_acceptance() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=(
                'Player movement, enemy behavior, auto fire, and collision '
                'have runtime smoke evidence.',
            )
        )
    )
    ledger.observe_source_change(('src/game.ts',), source_revision=1)
    ledger.observe_verification(
        VerificationEvidence(
            command='npm run build',
            cwd='.',
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            workspace_revision=1,
            source_revision=1,
            verification_type='build',
        )
    )

    assert ledger.criteria['criterion-1'].status != 'satisfied'


def test_unknown_domain_acceptance_uses_declared_evidence_type() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(
            criteria=(
                'Orbital telemetry reconciliation has runtime smoke evidence.',
            )
        )
    )
    ledger.observe_source_change(('src/orbit.py',), source_revision=1)
    ledger.observe_verification(
        VerificationEvidence(
            command='python -m compileall src',
            cwd='.',
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            workspace_revision=1,
            source_revision=1,
            verification_type='build',
        )
    )
    assert ledger.criteria['criterion-1'].status == 'partially_satisfied'

    ledger.observe_verification(
        VerificationEvidence(
            command='python -m pytest tests/smoke',
            cwd='.',
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
            workspace_revision=1,
            source_revision=1,
            verification_type='smoke',
        )
    )

    assert ledger.criteria['criterion-1'].status == 'satisfied'


def test_finish_task_gap_report_names_missing_criterion(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.begin_turn('Implement behavior')
    contract = change_contract(
        criteria=('Runtime behavior has smoke evidence.',)
    )
    ledger = AcceptanceLedger.from_contract(contract)
    checker = CompletionChecker(None, None, manager, acceptance_ledger=ledger)
    checker.task_contract = contract
    finish = ToolResult.ok(
        'Declared change task completed.',
        metadata={
            'finish_task': True,
            'task_kind': 'change',
            'status': 'completed',
            'summary': 'Done.',
        },
    )

    reasons = asyncio.run(
        checker.finish_rejection_reasons(
            finish,
            working_state=object(),  # type: ignore[arg-type]
            mutation_attempted=True,
            change_required=True,
            verification=None,
        )
    )

    assert reasons
    report = checker.last_finish_gap_report
    assert report['missing_criteria'] == (
        'Runtime behavior has smoke evidence.',
    )
    assert 'Completion evidence is insufficient' in reasons[-1]


def test_read_only_contract_has_no_code_acceptance() -> None:
    contract = TaskContract(
        intent=TurnIntent('inspect', 'high', 'inspect'),
        requires_change=False,
        requires_plan=False,
        completion_contract='inspection',
        initial_phase='exploring',
        initial_tool_surface='read_only',
        deliverables=('repository-grounded analysis',),
        acceptance_criteria=(
            'The answer is grounded in repository evidence.',
            'No workspace edit is made.',
        ),
    )
    ledger = AcceptanceLedger.from_contract(contract)

    assert all('smoke' not in item.casefold() for item in contract.acceptance_criteria)
    assert ledger.pending_ids() == ('criterion-1', 'criterion-2')


def test_session_rollout_replays_acceptance_ledger_state(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    ledger = AcceptanceLedger.from_contract(
        change_contract(criteria=('Typecheck passes.',))
    )
    ledger.record_evidence(
        AcceptanceEvidence(
            criterion_id='criterion-1',
            criterion_text='Typecheck passes.',
            status='satisfied',
            evidence_type='typecheck',
            verification_record_ids=('tsc:typecheck:1',),
            source_revision=1,
            producer='test',
            confidence=0.95,
            explanation='Typecheck passed.',
        )
    )
    snapshot = store.record_event(
        AcceptanceLedgerUpdated(
            evidence=tuple(
                item.as_dict()
                for item in ledger.evidence_snapshot(
                    criterion_ids=('criterion-1',)
                )
            ),
            completed_criterion_ids=('criterion-1',),
            source_revision=1,
            ledger=ledger.as_dict(),
        ),
        [{'role': 'user', 'content': 'fix it'}],
        session_id=None,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )

    resumed = store.load(snapshot.id)

    assert resumed.acceptance_ledger is not None
    rebuilt = AcceptanceLedger.from_dict(resumed.acceptance_ledger)
    assert rebuilt.satisfied_ids() == ('criterion-1',)


def test_acceptance_level_requires_every_criterion_to_be_satisfied() -> None:
    ledger = AcceptanceLedger.from_contract(
        change_contract(criteria=('Typecheck passes.', 'Unit tests pass.'))
    )
    ledger.record_evidence(
        AcceptanceEvidence(
            criterion_id='criterion-1',
            criterion_text='Typecheck passes.',
            status='satisfied',
            evidence_type='typecheck',
            producer='test',
            confidence=1.0,
        )
    )

    assert ledger.acceptance_verified is False

    ledger.record_evidence(
        AcceptanceEvidence(
            criterion_id='criterion-2',
            criterion_text='Unit tests pass.',
            status='satisfied',
            evidence_type='test_result',
            producer='test',
            confidence=1.0,
        )
    )

    assert ledger.acceptance_verified is True
