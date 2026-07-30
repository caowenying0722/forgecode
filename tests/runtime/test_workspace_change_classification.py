'''Cross-tool workspace classification and verification revision tests.'''

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from forge.context.working import WorkingState
from forge.runtime.completion import CompletionGate, TaskPolicy
from forge.runtime.completion_checker import CompletionChecker
from forge.runtime.acceptance import AcceptanceLedger
from forge.runtime.intent import TaskContract, TurnIntent, VerificationPolicy
from forge.runtime.state import VerificationEvidence
from forge.runtime.verification import verification_artifact_scope
from forge.runtime.workspace import WorkspaceTracker
from forge.runtime.workspace_classification import (
    ArtifactRule,
    VerificationArtifactScope,
)
from forge.tasks.manager import TaskManager
from forge.tools.base import ToolResult
from forge.tools.verify import VerifyTool


def run(coroutine: object):
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def initialize_git_repository(root: Path) -> None:
    subprocess.run(['git', 'init', '--quiet'], cwd=root, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'forge@example.test'],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'ForgeCode Tests'],
        cwd=root,
        check=True,
    )
    (root / 'sample.txt').write_text('baseline\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'baseline'],
        cwd=root,
        check=True,
    )


def evidence(
    source_revision: int,
    *,
    command: str = 'npx tsc --noEmit -p tsconfig.json && npx vite build',
    verification_type: str = 'build',
) -> VerificationEvidence:
    return VerificationEvidence(
        command=command,
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=source_revision,
        source_revision=source_revision,
        verification_type=verification_type,
    )


def contract_with_criteria(criteria: tuple[str, ...]) -> TaskContract:
    return TaskContract(
        intent=TurnIntent('implement', 'high', 'test contract'),
        requires_change=True,
        requires_plan=False,
        completion_contract='change',
        initial_phase='implementing',
        initial_tool_surface='all',
        deliverables=('workspace changes',),
        acceptance_criteria=criteria,
        verification_policy=VerificationPolicy(kind='required', required=True),
    )


def test_typescript_vite_artifacts_advance_filesystem_not_source_revision(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'tsconfig.json').write_text('{}\n', encoding='utf-8')
    source = tmp_path / 'src' / 'main.ts'
    source.parent.mkdir()
    source.write_text('export const oldValue = 1;\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'ts'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    source.write_text('export const newValue = 2;\n', encoding='utf-8')
    source_change = run(tracker.refresh())
    current = evidence(tracker.source_revision)
    output = tmp_path / 'dist' / 'assets' / 'index-a1b2c3.js'
    output.parent.mkdir(parents=True)
    output.write_text('console.log("bundle")\n', encoding='utf-8')
    artifact_change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                current.command,
                root=tmp_path,
                target='build',
            ),
        )
    )
    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            current,
            mutation_attempted=True,
        )
    )

    assert source_change is not None and source_change.source_revision == 1
    assert artifact_change is not None
    assert artifact_change.filesystem_revision == 2
    assert artifact_change.source_revision == 1
    assert tracker.source_revision == 1
    assert current.success is True
    assert decision.allowed is True


def test_typescript_emit_to_src_is_verification_side_effect_not_progress(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    source = tmp_path / 'src' / 'app.ts'
    source.parent.mkdir()
    source.write_text('export const value = 1;\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'ts'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    (tmp_path / 'src' / 'app.js').write_text('export const value = 1;\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx tsc -p tsconfig.json',
                root=tmp_path,
                target='typecheck',
            ),
        )
    )

    assert change is not None
    assert change.classification.verification_side_effect_paths == ('src/app.js',)
    assert tracker.source_revision == 0
    assert tracker.changed_paths == ()


def test_existing_javascript_source_change_advances_source_revision(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    source = tmp_path / 'src' / 'app.js'
    source.parent.mkdir()
    source.write_text('export const value = 1;\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'js'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    source.write_text('export const value = 2;\n', encoding='utf-8')
    change = run(tracker.refresh())

    assert change is not None
    assert change.source_paths == ('src/app.js',)
    assert tracker.source_revision == 1


def test_pytest_cache_does_not_stale_verification(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='python -m pytest -q', verification_type='test')

    cache = tmp_path / '.pytest_cache' / 'v' / 'cache' / 'nodeids'
    cache.parent.mkdir(parents=True)
    cache.write_text('[]\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                current.command,
                root=tmp_path,
                target='test',
            ),
        )
    )

    assert change is not None
    assert change.classification.cache_paths == ('.pytest_cache/v/cache/nodeids',)
    assert tracker.source_revision == current.bound_source_revision
    assert run(CompletionGate(tmp_path).evaluate(tracker, current, mutation_attempted=False)).allowed


def test_coverage_output_is_not_task_progress(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    report = tmp_path / 'htmlcov' / 'index.html'
    report.parent.mkdir()
    report.write_text('<html></html>\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'coverage run -m pytest && coverage html',
                root=tmp_path,
                target='test',
            ),
        )
    )

    assert change is not None
    assert change.classification.generated_paths == ('htmlcov/index.html',)
    assert change.source_paths == ()


def test_cargo_target_does_not_stale_verification(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='cargo test', verification_type='test')

    artifact = tmp_path / 'target' / 'debug' / 'deps' / 'app.rlib'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'compiled')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope('cargo test', root=tmp_path, target='test'),
        )
    )

    assert change is not None
    assert change.source_revision == 0
    assert run(CompletionGate(tmp_path).evaluate(tracker, current, mutation_attempted=False)).allowed


def test_gradle_build_output_does_not_stale_verification(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='gradle test', verification_type='test')

    artifact = tmp_path / 'build' / 'classes' / 'java' / 'main' / 'App.class'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'bytecode')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope('gradle test', root=tmp_path, target='test'),
        )
    )

    assert change is not None
    assert change.source_revision == 0
    assert run(CompletionGate(tmp_path).evaluate(tracker, current, mutation_attempted=False)).allowed


def test_verification_source_side_effects_are_invalid_for_handwritten_sources(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    for relative in ('src/app.ts', 'app.py', 'src/lib.rs', 'src/App.java'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'sources'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    for relative in ('src/app.ts', 'app.py', 'src/lib.rs', 'src/App.java'):
        (tmp_path / relative).write_text('new\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=VerificationArtifactScope(),
        )
    )

    assert change is not None
    assert set(change.classification.verification_side_effect_paths) == {
        'app.py',
        'src/App.java',
        'src/app.ts',
        'src/lib.rs',
    }
    assert change.source_revision == 1


def test_only_generated_artifacts_do_not_require_reverification(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='npx vite build', verification_type='build')

    bundle = tmp_path / 'dist' / 'bundle-9f8e7d.js'
    bundle.parent.mkdir()
    bundle.write_text('bundle\n', encoding='utf-8')
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope('npx vite build', root=tmp_path, target='build'),
        )
    )

    assert run(CompletionGate(tmp_path, TaskPolicy(require_verification=True)).evaluate(tracker, current, mutation_attempted=False)).allowed


def test_source_change_makes_existing_verification_stale(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='python -m pytest -q', verification_type='test')

    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())
    decision = run(CompletionGate(tmp_path, TaskPolicy(require_verification=True)).evaluate(tracker, current, mutation_attempted=False))

    assert decision.allowed is False
    assert any('changed after verification' in reason for reason in decision.reasons)


def test_same_source_revision_reuses_successful_verification(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    tool = VerifyTool(tmp_path, tracker)

    first = run(tool.run({'target': 'diff'}))
    second = run(tool.run({'target': 'diff'}))

    assert first.success is True
    assert second.success is True
    assert second.metadata['verification_reused'] is True
    assert second.metadata['source_revision'] == first.metadata['source_revision']


def test_related_source_and_unrelated_temp_are_reported_separately(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    source = tmp_path / 'src' / 'app.py'
    source.parent.mkdir()
    source.write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'source'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    source.write_text('new\n', encoding='utf-8')
    (tmp_path / 'tmp-note.txt').write_text('temp\n', encoding='utf-8')
    change = run(tracker.refresh())

    assert change is not None
    assert change.source_paths == ('src/app.py',)
    assert change.classification.unrelated_paths == ('tmp-note.txt',)


def test_forbidden_file_is_not_hidden_by_related_change(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    hidden = tmp_path / 'tests' / 'hidden' / 'test_secret.py'
    hidden.parent.mkdir(parents=True)
    hidden.write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'hidden'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    hidden.write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())
    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            evidence(tracker.source_revision, command='python -m pytest -q', verification_type='test'),
            mutation_attempted=True,
        )
    )

    assert decision.allowed is False
    assert any('Forbidden paths' in reason for reason in decision.reasons)


def test_finish_rejects_config_only_build_without_smoke_evidence(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'package.json').write_text('{"scripts":{"build":"vite build"}}\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    run(tracker.refresh())
    task_manager = TaskManager(tmp_path)
    task_manager.begin_turn('帮我优先实现“三选一升级 + 武器组合 + Boss”')
    contract = contract_with_criteria(
        ('Runtime behavior has smoke evidence.',)
    )
    ledger = AcceptanceLedger.from_contract(contract)
    ledger.observe_verification(
        evidence(
            tracker.source_revision,
            command='npx vite build',
            verification_type='build',
        )
    )
    checker = CompletionChecker(
        tracker,
        CompletionGate(tmp_path),
        task_manager,
        acceptance_ledger=ledger,
    )
    checker.task_contract = contract
    finish = ToolResult.ok(
        'Declared change task completed.',
        metadata={
            'finish_task': True,
            'task_kind': 'change',
            'status': 'completed',
            'summary': 'Updated config and build passes.',
        },
    )

    reasons = run(
        checker.finish_rejection_reasons(
            finish,
            working_state=WorkingState(),
            mutation_attempted=True,
            change_required=True,
            verification=evidence(tracker.source_revision, command='npx vite build', verification_type='build'),
        )
    )

    assert reasons
    assert checker.last_finish_gap_report['missing_criteria'] == (
        'Runtime behavior has smoke evidence.',
    )
    assert checker.last_finish_gap_report['missing_verification'] == ()


def test_finish_accepts_when_acceptance_ledger_has_evidence(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    scene = tmp_path / 'src' / 'game' / 'PlayScene.ts'
    scene.parent.mkdir(parents=True)
    scene.write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '--quiet', '-m', 'scene'], cwd=tmp_path, check=True)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    scene.write_text(
        'class PlayScene { create() { this.showThreeUpgradeChoices(); '
        'this.weaponComboTrigger(); this.spawnBossByWave(); } '
        'showThreeUpgradeChoices() { return ["upgrade option", "upgrade option", "upgrade option"]; } '
        'selectUpgradeOption() { this.applyUpgradeToPlayerWeaponState(); } '
        'weaponComboTrigger() { this.activateComboCombination(); } '
        'spawnBossByWave() { return { bossHealth: 100, bossPhaseBehavior: 1 }; } }\n',
        encoding='utf-8',
    )
    run(tracker.refresh())
    task_manager = TaskManager(tmp_path)
    task_manager.begin_turn('帮我优先实现“三选一升级 + 武器组合 + Boss”')
    contract = contract_with_criteria(
        (
            'A source diff exists.',
            'Runtime behavior has smoke evidence.',
        )
    )
    ledger = AcceptanceLedger.from_contract(contract)
    ledger.observe_source_change(('src/game/PlayScene.ts',), source_revision=1)
    ledger.observe_verification(
        evidence(
            tracker.source_revision,
            command='npm run smoke',
            verification_type='smoke',
        )
    )
    checker = CompletionChecker(
        tracker,
        CompletionGate(tmp_path),
        task_manager,
        acceptance_ledger=ledger,
    )
    checker.task_contract = contract
    finish = ToolResult.ok(
        'Declared change task completed.',
        metadata={
            'finish_task': True,
            'task_kind': 'change',
            'status': 'completed',
            'summary': 'Implemented runtime systems.',
        },
    )

    reasons = run(
        checker.finish_rejection_reasons(
            finish,
            working_state=WorkingState(),
            mutation_attempted=True,
            change_required=True,
            verification=evidence(tracker.source_revision),
            evidence_paths=('src/game/PlayScene.ts',),
        )
    )

    assert reasons == ()


def test_build_artifact_loop_does_not_appear_as_stale_build_cycle(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    current = evidence(0, command='npx vite build', verification_type='build')

    for name in ('app-a1.js', 'app-b2.js'):
        bundle = tmp_path / 'dist' / name
        bundle.parent.mkdir(exist_ok=True)
        bundle.write_text(name, encoding='utf-8')
        run(
            tracker.refresh(
                origin='verification',
                artifact_scope=verification_artifact_scope('npx vite build', root=tmp_path, target='build'),
            )
        )
        decision = run(
            CompletionGate(tmp_path, TaskPolicy(require_verification=True)).evaluate(
                tracker,
                current,
                mutation_attempted=False,
            )
        )
        assert decision.allowed is True


def test_unknown_build_tool_adapter_can_be_registered_without_core_changes(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    scope = VerificationArtifactScope(
        verification_type='build',
        allowed_writes=(
            ArtifactRule('.unknown-out/**', 'generated_artifact', 'plugin output'),
        ),
    )

    artifact = tmp_path / '.unknown-out' / 'result.bin'
    artifact.parent.mkdir()
    artifact.write_bytes(b'ok')
    change = run(tracker.refresh(origin='verification', artifact_scope=scope))

    assert change is not None
    assert change.classification.generated_paths == ('.unknown-out/result.bin',)
    assert change.source_revision == 0
