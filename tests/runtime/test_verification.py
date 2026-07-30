'''Tests for project validation command discovery.'''

import asyncio
from pathlib import Path
import subprocess

from forge.runtime.verification import (
    choose_validation_command,
    classify_verification_command,
    discover_validation_commands,
)
from forge.runtime.verification_ledger import VerificationLedger
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.shell import RunCommandTool


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
