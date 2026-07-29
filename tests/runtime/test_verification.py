'''Tests for project validation command discovery.'''

from pathlib import Path

from forge.runtime.verification import (
    choose_validation_command,
    classify_verification_command,
    discover_validation_commands,
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
