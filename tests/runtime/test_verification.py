'''Tests for project validation command discovery.'''

import asyncio
from pathlib import Path
import subprocess

import pytest

import forge.runtime.verification as verification_module
from forge.runtime.process import ProcessResult
from forge.runtime.completion_checker import verification_from_result
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
from forge.tools.base import ToolResult


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


def test_verification_ledger_preserves_artifact_deltas() -> None:
    ledger = VerificationLedger()

    record = ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'command': 'npm run build',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'artifact_deltas': [
                {
                    'path': 'dist/assets/index-OLD.js',
                    'operation': 'deleted',
                    'kind': 'generated_artifact',
                    'before_fingerprint': 'file:old',
                    'after_fingerprint': None,
                    'rule_pattern': 'dist/**',
                    'rule_reason': 'vite build output',
                }
            ],
        }
    )

    assert record is not None
    assert record.artifact_deltas[0].operation == 'deleted'
    assert record.artifact_deltas[0].before_fingerprint == 'file:old'
    assert record.to_evidence().artifact_deltas == record.artifact_deltas


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
            'artifact_deltas': [
                {
                    'path': 'dist/assets/index-OLD.js',
                    'operation': 'deleted',
                    'kind': 'generated_artifact',
                    'before_fingerprint': 'file:old',
                    'after_fingerprint': None,
                    'rule_pattern': 'dist/**',
                    'rule_reason': 'vite build output',
                }
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
    assert cached.artifact_deltas[0].operation == 'deleted'


def test_verification_from_result_preserves_artifact_deltas() -> None:
    result = ToolResult.ok(
        'passed',
        metadata={
            'verification': True,
            'verification_status': 'passed',
            'command': 'npm run build',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'artifact_deltas': [
                {
                    'path': 'dist/assets/index-OLD.js',
                    'operation': 'deleted',
                    'kind': 'generated_artifact',
                    'before_fingerprint': 'file:old',
                    'after_fingerprint': None,
                    'rule_pattern': 'dist/**',
                    'rule_reason': 'vite build output',
                }
            ],
        },
    )

    evidence = verification_from_result(result)

    assert evidence is not None
    assert evidence.artifact_deltas[0].path == 'dist/assets/index-OLD.js'
    assert evidence.artifact_deltas[0].operation == 'deleted'


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

    monkeypatch.setattr(
        'forge.runtime.verification_executor.run_process',
        clean_process,
    )
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

    monkeypatch.setattr(
        'forge.runtime.verification_executor.run_process',
        clean_process,
    )
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

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
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

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
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

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
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


def _write_package_json(root: Path, scripts: dict[str, str]) -> None:
    import json

    (root / 'package.json').write_text(
        json.dumps({'scripts': scripts}),
        encoding='utf-8',
    )


def test_explicit_npm_build_resolves_vite_artifact_scope(
    tmp_path: Path,
) -> None:
    _write_package_json(
        tmp_path,
        {'build': 'tsc --noEmit && vite build'},
    )

    resolved = verification_module.resolve_project_command(
        'npm run build',
        tmp_path,
    )

    assert resolved.invoked_command == 'npm run build'
    assert resolved.effective_commands == ('tsc --noEmit', 'vite build')
    assert resolved.verification_types == ('typecheck', 'build')
    assert 'dist/**' in resolved.artifact_scope.allowed_write_patterns


def test_npm_test_resolves_test_script(tmp_path: Path) -> None:
    _write_package_json(tmp_path, {'test': 'vitest run'})

    resolved = verification_module.resolve_project_command('npm test', tmp_path)
    explicit_run = verification_module.resolve_project_command(
        'npm run test',
        tmp_path,
    )

    assert resolved.effective_commands == ('vitest run',)
    assert explicit_run.effective_commands == resolved.effective_commands
    assert resolved.verification_types == ('test',)
    assert 'coverage/**' in resolved.artifact_scope.allowed_write_patterns


def _assert_package_manager_build_resolves(
    tmp_path: Path,
    command: str,
) -> None:
    _write_package_json(tmp_path, {'build': 'vite build'})

    resolved = verification_module.resolve_project_command(command, tmp_path)

    assert resolved.invoked_command == command
    assert resolved.effective_commands == ('vite build',)
    assert resolved.verification_types == ('build',)
    assert 'dist/**' in resolved.artifact_scope.allowed_write_patterns


def test_pnpm_build_resolves_package_script(tmp_path: Path) -> None:
    _assert_package_manager_build_resolves(tmp_path, 'pnpm build')
    _assert_package_manager_build_resolves(tmp_path, 'pnpm run build')


def test_yarn_build_resolves_package_script(tmp_path: Path) -> None:
    _assert_package_manager_build_resolves(tmp_path, 'yarn run build')
    _assert_package_manager_build_resolves(tmp_path, 'yarn build')


def test_bun_run_build_resolves_package_script(tmp_path: Path) -> None:
    _assert_package_manager_build_resolves(tmp_path, 'bun run build')


def test_nested_package_scripts_are_resolved(tmp_path: Path) -> None:
    _write_package_json(
        tmp_path,
        {'build': 'npm run compile', 'compile': 'vite build'},
    )

    resolved = verification_module.resolve_project_command(
        'npm run build',
        tmp_path,
    )

    assert resolved.effective_commands == ('vite build',)
    assert resolved.script_chain == (
        'npm run build',
        'npm run compile',
        'vite build',
    )
    assert 'dist/**' in resolved.artifact_scope.allowed_write_patterns


def test_package_script_lifecycle_hooks_are_resolved_in_order(
    tmp_path: Path,
) -> None:
    _write_package_json(
        tmp_path,
        {
            'prebuild': 'eslint src',
            'build': 'vite build',
            'postbuild': 'vitest run',
        },
    )

    resolved = verification_module.resolve_project_command(
        'npm run build',
        tmp_path,
    )

    assert resolved.effective_commands == (
        'eslint src',
        'vite build',
        'vitest run',
    )
    assert resolved.script_chain == (
        'npm run build',
        'npm run prebuild',
        'eslint src',
        'vite build',
        'npm run postbuild',
        'vitest run',
    )
    assert resolved.verification_types == ('lint', 'build', 'test')


def test_recursive_package_script_is_rejected(tmp_path: Path) -> None:
    _write_package_json(
        tmp_path,
        {'build': 'npm run compile', 'compile': 'npm run build'},
    )

    with pytest.raises(Exception) as caught:
        verification_module.resolve_project_command('npm run build', tmp_path)

    assert getattr(caught.value, 'code', '') == 'recursive_package_script'
    assert 'build' in str(caught.value)
    assert 'compile' in str(caught.value)


def test_missing_package_script_is_reported(tmp_path: Path) -> None:
    _write_package_json(tmp_path, {'test': 'vitest run'})

    with pytest.raises(Exception) as caught:
        verification_module.resolve_project_command('npm run build', tmp_path)

    assert getattr(caught.value, 'code', '') == 'missing_package_script'


def test_invalid_package_json_is_reported_structurally(tmp_path: Path) -> None:
    (tmp_path / 'package.json').write_text('{invalid', encoding='utf-8')

    with pytest.raises(Exception) as caught:
        verification_module.resolve_project_command('npm run build', tmp_path)

    assert getattr(caught.value, 'code', '') == 'invalid_package_json'
    assert getattr(caught.value, 'path', None) == tmp_path / 'package.json'


def test_explicit_npm_build_and_auto_build_have_equivalent_artifact_scope(
    tmp_path: Path,
) -> None:
    _write_package_json(
        tmp_path,
        {'build': 'tsc --noEmit && vite build'},
    )
    (tmp_path / 'tsconfig.json').write_text('{}', encoding='utf-8')
    automatic = choose_validation_command(tmp_path, target='build')
    assert automatic is not None

    explicit = verification_module.resolve_project_command(
        'npm run build',
        tmp_path,
    )
    automatic_scope = verification_artifact_scope(
        automatic.command,
        root=tmp_path,
        target='build',
    )

    assert explicit.artifact_scope.verification_type == (
        automatic_scope.verification_type
    )
    assert explicit.artifact_scope.allowed_writes == automatic_scope.allowed_writes


def test_verify_and_promoted_run_command_share_execution_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()
    from forge.runtime.verification_executor import VerificationExecutor

    calls: list[tuple[type[object], str]] = []

    async def execute(
        self: object,
        *,
        command: str,
        **_kwargs: object,
    ) -> ToolResult:
        calls.append((type(self), command))
        return ToolResult.ok(
            'verification passed',
            metadata={
                'verification': True,
                'verification_status': 'passed',
                'command': command,
            },
        )

    monkeypatch.setattr(VerificationExecutor, 'execute', execute)
    verify_result = run(
        VerifyTool(tmp_path, tracker, ledger).run(
            {'command': 'git diff --check'}
        )
    )
    run_result = run(
        RunCommandTool(
            tmp_path,
            workspace_tracker=tracker,
            verification_ledger=ledger,
        ).run({'command': 'git diff --check'})
    )

    assert verify_result.success is True
    assert run_result.success is True
    assert calls == [
        (VerificationExecutor, 'git diff --check'),
        (VerificationExecutor, 'git diff --check'),
    ]


def test_verification_cache_is_cleared_at_begin_turn(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    tracker.verification_cache[('old-turn',)] = object()

    run(tracker.begin_turn())

    assert tracker.verification_cache == {}


def test_same_relative_revision_in_new_turn_does_not_reuse_old_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    ledger = VerificationLedger()
    calls = 0

    async def process(*_args: object, **_kwargs: object) -> ProcessResult:
        nonlocal calls
        calls += 1
        return _successful_process()

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
    tool = VerifyTool(tmp_path, tracker, ledger)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('turn a\n', encoding='utf-8')
    run(tracker.refresh())
    first = run(tool.run({'command': 'git diff --check'}))

    run(tracker.begin_turn())
    ledger.clear_turn()
    (tmp_path / 'sample.txt').write_text('turn b\n', encoding='utf-8')
    run(tracker.refresh())
    second = run(tool.run({'command': 'git diff --check'}))

    assert first.metadata['source_revision'] == 1
    assert second.metadata['source_revision'] == 1
    assert first.metadata.get('verification_reused') is not True
    assert second.metadata.get('verification_reused') is not True
    assert calls == 2


def test_same_turn_same_snapshot_can_reuse_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    ledger = VerificationLedger()
    calls = 0

    async def process(*_args: object, **_kwargs: object) -> ProcessResult:
        nonlocal calls
        calls += 1
        return _successful_process()

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
    tool = VerifyTool(tmp_path, tracker, ledger)
    first = run(tool.run({'command': 'git diff --check'}))
    second = run(tool.run({'command': 'git diff --check'}))

    assert first.success is True
    assert second.success is True
    assert second.metadata['verification_reused'] is True
    assert calls == 1


def test_cached_verification_rechecks_artifact_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_git_repository(tmp_path)
    _write_package_json(tmp_path, {'build': 'vite build'})
    source = tmp_path / 'src' / 'main.ts'
    source.parent.mkdir()
    source.write_text('export const value = 1;\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'vite project'],
        cwd=tmp_path,
        check=True,
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    source.write_text('export const value = 2;\n', encoding='utf-8')
    run(tracker.refresh())
    ledger = VerificationLedger()
    calls = 0

    async def process(*_args: object, **_kwargs: object) -> ProcessResult:
        nonlocal calls
        calls += 1
        output = tmp_path / 'dist' / 'index.html'
        output.parent.mkdir(exist_ok=True)
        output.write_text(f'built {calls}\n', encoding='utf-8')
        return _successful_process()

    monkeypatch.setattr('forge.runtime.verification_executor.run_process', process)
    tool = VerifyTool(tmp_path, tracker, ledger)
    first = run(tool.run({'command': 'npm run build', 'target': 'build'}))
    (tmp_path / 'dist' / 'index.html').write_text(
        'tampered\n',
        encoding='utf-8',
    )
    run(tracker.refresh())
    second = run(tool.run({'command': 'npm run build', 'target': 'build'}))

    assert first.success is True
    assert second.success is True
    assert second.metadata.get('verification_reused') is not True
    assert calls == 2
    assert (tmp_path / 'dist' / 'index.html').read_text(encoding='utf-8') == (
        'built 2\n'
    )
