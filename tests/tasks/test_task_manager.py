'''Tests for current-task anchoring and optional persistent plans.'''

from pathlib import Path

import pytest

from forge.tasks.manager import TaskManager
from forge.tasks.state import SourceSection, TaskSpecDigest


def test_simple_task_stays_in_memory_without_creating_files(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)

    task = manager.start('Can you see the play directory?')

    assert task.planned is False
    assert 'Can you see the play directory?' in manager.system_suffix()
    assert not (tmp_path / '.forge' / 'tasks').exists()


def test_complex_plan_persists_updates_and_resumes(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Fix all six block faces and verify the game.')

    planned = manager.plan(
        ['Inspect geometry', 'Fix UVs', 'Verify'],
        constraints=['Focus on play'],
        scope_hints=['play/**'],
    )
    updated = manager.update_step(
        'step-1',
        'completed',
        evidence=['Read play/js/world.js'],
    )

    assert planned.planned is True
    assert updated.current_step_id == 'step-2'
    assert updated.steps[0].evidence == ('Read play/js/world.js',)
    assert (tmp_path / '.forge' / 'tasks' / f'{planned.id}.json').exists()

    restarted = TaskManager(tmp_path)
    resumed = restarted.resume(planned.id)

    assert resumed.goal == planned.goal
    assert resumed.current_step_id == 'step-2'
    assert 'Fix UVs' in restarted.system_suffix()

    continued = restarted.begin_turn('Continue from the saved task')
    following = restarted.begin_turn('Start a separate task')

    assert continued.id == planned.id
    assert continued.goal == planned.goal
    assert following.id != planned.id
    assert following.goal == 'Start a separate task'


def test_plan_is_optional_and_cannot_be_recreated_accidentally(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Complex task')
    manager.plan(['Inspect', 'Implement'])

    with pytest.raises(ValueError, match='already has a plan'):
        manager.plan(['Start over', 'Finish'])


def test_completion_and_blocking_are_persisted_for_planned_tasks(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    task = manager.start('Implement and verify')
    manager.plan(['Implement', 'Verify'])

    blocked = manager.block(('Verification failed.',))

    assert blocked is not None and blocked.status == 'blocked'
    assert manager.store.load(task.id).blocked_reasons == (
        'Verification failed.',
    )

    manager.resume(task.id)
    completed = manager.complete()

    assert completed is not None and completed.status == 'completed'
    assert manager.store.load(task.id).status == 'completed'


def test_followup_after_stuck_keeps_root_goal_and_latest_directive(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    original = manager.start('Fix the rendering bug in play/js/world.js')
    manager.stuck(('Repeated actions did not make progress.',))

    continued = manager.begin_turn('你直接帮我修复')

    assert continued.id == original.id
    assert continued.goal == original.goal
    assert continued.status == 'in_progress'
    suffix = manager.system_suffix()
    assert original.goal in suffix
    assert '你直接帮我修复' in suffix


def test_resume_rejects_invalid_task_id(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)

    with pytest.raises(ValueError, match='Invalid task ID'):
        manager.resume('../../outside')


def test_continue_next_step_does_not_rebuild_existing_plan(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    original = manager.start('Deliver the requested change')
    manager.plan(['Inspect', 'Implement', 'Verify'])
    manager.update_step('step-1', 'completed')
    manager.continue_next_turn()

    continued = manager.begin_turn('Continue the next step')

    assert continued.id == original.id
    assert continued.current_step_id == 'step-2'
    assert [step.title for step in continued.steps] == [
        'Inspect',
        'Implement',
        'Verify',
    ]


def test_task_spec_digest_preserves_acceptance_criteria() -> None:
    digest = TaskSpecDigest(
        source_paths=('docs/task.md',),
        goal='Implement the documented behavior',
        requirements=('Keep existing public APIs compatible.',),
        acceptance_criteria=(
            'The focused unit suite passes.',
            'The changed behavior has revision-bound evidence.',
        ),
        required_commands=('pytest -q tests/unit',),
        required_modules=('forge/runtime',),
        forbidden_changes=('tests/fixtures/frozen/**',),
    )

    restored = TaskSpecDigest.from_dict(digest.as_dict())

    assert restored.acceptance_criteria == digest.acceptance_criteria
    assert restored.required_commands == ('pytest -q tests/unit',)


def test_task_spec_digest_can_link_back_to_source_lines() -> None:
    section = SourceSection(
        path='docs/task.md',
        start_line=41,
        end_line=58,
        title='Acceptance',
    )
    digest = TaskSpecDigest(
        source_paths=('docs/task.md',),
        goal='Apply the task specification',
        relevant_sections=(section,),
    )

    rendered = digest.render()

    assert 'docs/task.md:41-58' in rendered
    assert 'authoritative source' in rendered.lower()


def test_resume_context_renders_changed_paths_and_blockers(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Continue structured work')
    manager.set_resume_context(
        {
            'changed_paths': ['forge/runtime/process.py'],
            'latest_failure': 'failed: typecheck (exit 1)',
        }
    )
    manager.block(('A required external service is unavailable.',))

    suffix = manager.system_suffix()

    assert 'Current changed paths:\n- forge/runtime/process.py' in suffix
    assert 'Latest failed diagnostic:\nfailed: typecheck' in suffix
    assert 'Current blockers:' in suffix
