'''Tests for project validation command discovery.'''

import asyncio
from pathlib import Path
import subprocess

import pytest

from forge.runtime.process import ProcessResult
from forge.runtime.verification import (
    choose_validation_command,
    classify_verification_command,
    discover_validation_commands,
    verification_artifact_scope,
)
from forge.runtime.verification_ledger import VerificationLedger
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.shell import RunCommandTool
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
    (root / 'sample.txt').write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'baseline'],
        cwd=root,
        check=True,
    )


def test_package_json_validation_discovery_prefers_project_scripts(
    tmp_path: Path,
) -> None:
    (tmp_path / 'package.json').write_text(
        '{"scripts":{"build":"vite build","test":"vitest run"}}\n',
        encoding='utf-8',
    )

    commands = discover_validation_commands(tmp_path)
    auto = choose_validation_command(tmp_path)
    build = choose_validation_command(tmp_path, target='build')

    assert {command.id for command in commands} >= {
        'npm:build',
        'npm:test',
        'diff',
    }
    assert auto is not None
    assert auto.id == 'npm:test'
    assert build is not None
    assert build.command == 'npm run build --if-present'


def test_typescript_vite_build_uses_no_emit_typecheck(
    tmp_path: Path,
) -> None:
    (tmp_path / 'tsconfig.json').write_text(
        '{"compilerOptions":{"strict":true}}\n',
        encoding='utf-8',
    )
    (tmp_path / 'package.json').write_text(
        '{"scripts":{"build":"tsc -p tsconfig.json && vite build"}}\n',
        encoding='utf-8',
    )

    build = choose_validation_command(tmp_path, target='build')

    assert build is not None
    assert build.command == 'npx tsc --noEmit -p tsconfig.json && npx vite build'


def test_validation_command_classifier_rejects_probe_and_dev_server(
    tmp_path: Path,
) -> None:
    commands = discover_validation_commands(tmp_path)

    probe_status, _ = classify_verification_command(
        'node -v && npm -v && pwd',
        discovered_commands=commands,
    )
    dev_status, _ = classify_verification_command(
        'npm run dev',
        discovered_commands=commands,
    )
    diff_status, _ = classify_verification_command(
        'git diff --check',
        discovered_commands=commands,
    )

    assert probe_status == 'invalid'
    assert dev_status == 'invalid'
    assert diff_status == 'passed'


def test_empty_package_json_does_not_satisfy_build_discovery(
    tmp_path: Path,
) -> None:
    (tmp_path / 'package.json').write_text('', encoding='utf-8')

    auto = choose_validation_command(tmp_path)
    build = choose_validation_command(tmp_path, target='build')

    assert auto is None
    assert build is None


def test_run_command_validation_routes_to_verification_ledger(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()
    tool = RunCommandTool(
        tmp_path,
        workspace_tracker=tracker,
        verification_ledger=ledger,
    )

    first = run(tool.execute(tool.input_model(command='git diff --check')))
    second = run(tool.execute(tool.input_model(command='git diff --check')))

    assert first.success is True
    assert first.metadata['verification'] is True
    assert first.metadata['verification_status'] == 'passed'
    assert ledger.latest_evidence(0) is not None
    assert ledger.latest_evidence(0).success is True  # type: ignore[union-attr]
    assert second.success is True
    assert second.metadata['verification_reused'] is True
    assert ledger.records[-1].evidence_source == 'cache'


def test_verification_ledger_preserves_generated_artifact_paths() -> None:
    ledger = VerificationLedger()

    record = ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'command': 'npx vite build',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'generated_artifact_paths': [
                'dist/index.html',
                'dist/assets/app.js',
            ],
        }
    )

    assert record is not None
    evidence = record.to_evidence()
    assert evidence.generated_artifact_paths == (
        'dist/index.html',
        'dist/assets/app.js',
    )


def test_verification_ledger_preserves_cache_paths() -> None:
    ledger = VerificationLedger()

    record = ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'command': 'python -m pytest -q',
            'cwd': '.',
            'workspace_revision': 0,
            'source_revision': 0,
            'filesystem_revision': 1,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'cache_paths': ['.pytest_cache/v/cache/nodeids'],
        }
    )

    assert record is not None
    assert record.to_evidence().cache_paths == (
        '.pytest_cache/v/cache/nodeids',
    )


def test_verification_ledger_preserves_artifact_fingerprints() -> None:
    ledger = VerificationLedger()

    record = ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'command': 'npx vite build',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'generated_artifact_paths': ['dist/index.html'],
            'cache_paths': ['.vite/cache.json'],
            'generated_artifact_fingerprints': [
                ['dist/index.html', 'file:abc']
            ],
            'cache_fingerprints': [['.vite/cache.json', 'file:def']],
        }
    )

    assert record is not None
    evidence = record.to_evidence()
    assert evidence.generated_artifact_fingerprints == (
        ('dist/index.html', 'file:abc'),
    )
    assert evidence.cache_fingerprints == (
        ('.vite/cache.json', 'file:def'),
    )


def test_cached_verification_preserves_artifact_evidence() -> None:
    ledger = VerificationLedger()
    key = ('source', 1, 'npx vite build')
    original = ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'command': 'npx vite build',
            'command_id': 'npm:build',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'generated_artifact_paths': ['dist/index.html'],
            'generated_artifact_fingerprints': [
                ['dist/index.html', 'file:abc']
            ],
        },
        reusable_key=key,
    )

    assert original is not None
    reusable = ledger.reusable(key)
    assert reusable is not None
    cached = reusable.to_evidence()
    assert cached.generated_artifact_paths == ('dist/index.html',)
    assert cached.generated_artifact_fingerprints == (
        ('dist/index.html', 'file:abc'),
    )


def _successful_process() -> ProcessResult:
    return ProcessResult(0, '', '', 0.01)


def _seed_verification_side_effect(
    root: Path,
    tracker: WorkspaceTracker,
) -> None:
    (root / 'rogue.log').write_text('unexpected\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx tsc --noEmit',
                root=root,
                target='typecheck',
            ),
        )
    )
    assert change is not None
    assert change.classification.verification_side_effect_paths == (
        'rogue.log',
    )


def test_verify_no_change_uses_empty_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    _seed_verification_side_effect(tmp_path, tracker)

    async def clean_process(*_args: object, **_kwargs: object) -> ProcessResult:
        return _successful_process()

    monkeypatch.setattr('forge.tools.verify.run_process', clean_process)
    result = run(
        VerifyTool(tmp_path, tracker).run({'command': 'git diff --check'})
    )

    assert result.success is True
    assert result.metadata['verification_side_effect_paths'] == []
    assert result.metadata['generated_artifact_paths'] == []
    assert result.metadata['cache_paths'] == []
    assert result.metadata['verification_transaction']['changed_paths'] == []


def test_run_command_no_change_uses_empty_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    _seed_verification_side_effect(tmp_path, tracker)
    ledger = VerificationLedger()

    async def clean_process(*_args: object, **_kwargs: object) -> ProcessResult:
        return _successful_process()

    monkeypatch.setattr('forge.tools.shell.run_process', clean_process)
    result = run(
        RunCommandTool(
            tmp_path,
            workspace_tracker=tracker,
            verification_ledger=ledger,
        ).run({'command': 'git diff --check'})
    )

    assert result.success is True
    assert result.metadata['verification_side_effect_paths'] == []
    assert result.metadata['generated_artifact_paths'] == []
    assert result.metadata['cache_paths'] == []
    assert result.metadata['verification_transaction']['changed_paths'] == []


def test_typecheck_after_build_does_not_inherit_build_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'package.json').write_text(
        '{"scripts":{"build":"tsc -p tsconfig.json && vite build"}}\n',
        encoding='utf-8',
    )
    (tmp_path / 'tsconfig.json').write_text('{}\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'typescript build'],
        cwd=tmp_path,
        check=True,
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()

    async def process(command: object, **_kwargs: object) -> ProcessResult:
        if 'vite build' in str(command):
            output = tmp_path / 'dist' / 'assets' / 'app.js'
            output.parent.mkdir(parents=True)
            output.write_text('bundle\n', encoding='utf-8')
        return _successful_process()

    monkeypatch.setattr('forge.tools.verify.run_process', process)
    tool = VerifyTool(tmp_path, tracker, ledger)
    build = run(tool.run({'target': 'build'}))
    typecheck = run(
        tool.run({'command': 'npx tsc --noEmit', 'target': 'typecheck'})
    )

    assert build.success is True
    assert build.metadata['generated_artifact_paths'] == [
        'dist/assets/app.js'
    ]
    assert typecheck.success is True
    assert typecheck.metadata['generated_artifact_paths'] == []
    assert typecheck.metadata['cache_paths'] == []
    assert typecheck.metadata['verification_side_effect_paths'] == []
    assert ledger.records[-2].generated_artifact_paths == (
        'dist/assets/app.js',
    )
    assert ledger.records[-1].generated_artifact_paths == ()


def test_failed_verification_does_not_contaminate_next_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()

    async def process(command: object, **_kwargs: object) -> ProcessResult:
        if command == 'npx tsc --noEmit':
            (tmp_path / 'rogue.log').write_text('failed output\n', encoding='utf-8')
            return ProcessResult(2, '', 'type error', 0.01)
        return _successful_process()

    monkeypatch.setattr('forge.tools.verify.run_process', process)
    tool = VerifyTool(tmp_path, tracker, ledger)
    failed = run(tool.run({'command': 'npx tsc --noEmit'}))
    clean = run(tool.run({'command': 'git diff --check'}))

    assert failed.success is False
    assert failed.metadata['verification_side_effect_paths'] == ['rogue.log']
    assert clean.success is True
    assert clean.metadata['verification_side_effect_paths'] == []
    assert ledger.records[-2].side_effect_paths == ('rogue.log',)
    assert ledger.records[-1].side_effect_paths == ()


def test_clean_command_does_not_reuse_previous_side_effect_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()

    async def process(command: object, **_kwargs: object) -> ProcessResult:
        if command == 'npx tsc --noEmit':
            (tmp_path / 'rogue.log').write_text('failed output\n', encoding='utf-8')
            return ProcessResult(2, '', 'type error', 0.01)
        return _successful_process()

    monkeypatch.setattr('forge.tools.shell.run_process', process)
    tool = RunCommandTool(
        tmp_path,
        workspace_tracker=tracker,
        verification_ledger=ledger,
    )
    failed = run(tool.run({'command': 'npx tsc --noEmit'}))
    clean = run(tool.run({'command': 'git diff --check'}))

    assert failed.success is False
    assert clean.success is True
    assert clean.metadata['verification_side_effect_paths'] == []
    assert ledger.records[-1].side_effect_paths == ()


def test_empty_workspace_delta_is_a_first_class_transaction(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    change = run(tracker.refresh(origin='verification'))

    assert change is not None
    assert change.paths == ()
    assert change.created_paths == ()
    assert change.modified_paths == ()
    assert change.deleted_paths == ()
    assert change.classification.changes == ()
    assert change.before_snapshot_id == change.after_snapshot_id
    assert tracker.filesystem_revision == 0
    assert tracker.source_revision == 0
