'''Project validation command discovery and verification classification.'''

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Literal


ValidationTarget = Literal['auto', 'build', 'test', 'lint', 'typecheck', 'diff']
VerificationStatus = Literal[
    'passed',
    'failed',
    'timed_out',
    'invalid',
    'unavailable',
]


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    id: str
    command: str
    cwd: str = '.'
    target: ValidationTarget = 'auto'
    source: str = 'discovered'
    strength: int = 0


NON_INTERACTIVE_ENV = {
    'CI': '1',
    'NO_COLOR': '1',
    'npm_config_yes': 'true',
    'npm_config_fund': 'false',
    'npm_config_audit': 'false',
    'GIT_TERMINAL_PROMPT': '0',
}


INTERACTIVE_COMMAND_PATTERN = re.compile(
    r'(?i)(?:'
    r'\b(?:read|pause)\b|'
    r'\b(?:vim|vi|nano|less|more)\b|'
    r'\b(?:serve|vite|next|webpack-dev-server)\b|'
    r'\b(?:npm|pnpm|yarn)\s+run\s+dev\b|'
    r'\b(?:npm|pnpm|yarn)\s+(?:start|dev)\b|'
    r'\bpython\s+-m\s+http\.server\b'
    r')'
)

PROBE_COMMAND_PATTERN = re.compile(
    r'(?i)(?:'
    r'\b(?:pwd|cd|dir|ls|tree|echo|type|cat)\b|'
    r'\b(?:node|npm|pnpm|yarn|python|py|pip|uv|cargo|go)\s+(?:-v|--version|version)\b|'
    r'\bgit\s+(?:status|log|show|branch|rev-parse)\b'
    r')'
)

VALIDATION_COMMAND_PATTERN = re.compile(
    r'(?i)(?:'
    r'\b(?:test|build|lint|typecheck|type-check|check)\b|'
    r'\b(?:tsc|vitest|jest|mocha|pytest|ruff|mypy|pyright|eslint)\b|'
    r'\b(?:cargo\s+(?:test|check|clippy)|go\s+test|dotnet\s+test|mvn\s+test|gradle\s+test)\b|'
    r'\bgit\s+diff\s+--check\b'
    r')'
)


def discover_validation_commands(root: Path) -> tuple[ValidationCommand, ...]:
    root = root.resolve()
    commands: list[ValidationCommand] = []
    commands.extend(_package_json_commands(root))
    commands.extend(_python_commands(root))
    commands.extend(_typescript_commands(root))
    commands.extend(_rust_commands(root))
    commands.extend(_go_commands(root))
    commands.append(
        ValidationCommand(
            id='diff',
            command='git diff --check',
            target='diff',
            source='git',
            strength=1,
        )
    )
    deduped: dict[str, ValidationCommand] = {}
    for command in commands:
        deduped.setdefault(command.id, command)
    return tuple(deduped.values())


def choose_validation_command(
    root: Path,
    *,
    target: ValidationTarget = 'auto',
    command_id: str = '',
) -> ValidationCommand | None:
    root = root.resolve()
    commands = discover_validation_commands(root)
    if command_id:
        return next((command for command in commands if command.id == command_id), None)
    if target != 'auto':
        targeted = [command for command in commands if command.target == target]
        if targeted:
            return max(targeted, key=lambda command: command.strength)
        if target != 'diff' and _has_project_validation_marker(root):
            return None
    candidates = [command for command in commands if command.target != 'diff']
    if not candidates and _has_project_validation_marker(root):
        return None
    if not candidates:
        candidates = list(commands)
    return max(candidates, key=lambda command: command.strength, default=None)


def classify_verification_command(
    command: str,
    *,
    discovered_commands: tuple[ValidationCommand, ...],
) -> tuple[VerificationStatus, str]:
    normalized = command.strip()
    if not normalized:
        return 'invalid', 'Verification command is empty.'
    if INTERACTIVE_COMMAND_PATTERN.search(normalized):
        return 'invalid', 'Verification command appears interactive or long-running.'
    if normalized in {item.command for item in discovered_commands}:
        return 'passed', ''
    if VALIDATION_COMMAND_PATTERN.search(normalized):
        return 'passed', ''
    if PROBE_COMMAND_PATTERN.search(normalized):
        return 'invalid', (
            'Verification command only inspects environment or files; it does '
            'not run a build, test, lint, type-check, or diff validation.'
        )
    return 'invalid', (
        'Verification command is not a recognized non-interactive project '
        'validation command.'
    )


def _package_json_commands(root: Path) -> list[ValidationCommand]:
    package_json = root / 'package.json'
    if not package_json.is_file():
        return []
    try:
        data = json.loads(package_json.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    scripts = data.get('scripts')
    if not isinstance(scripts, dict):
        return []
    commands: list[ValidationCommand] = []
    for script, target, strength in (
        ('test', 'test', 90),
        ('build', 'build', 85),
        ('typecheck', 'typecheck', 80),
        ('type-check', 'typecheck', 80),
        ('lint', 'lint', 75),
    ):
        value = scripts.get(script)
        if isinstance(value, str) and value.strip():
            commands.append(
                ValidationCommand(
                    id=f'npm:{script}',
                    command=f'npm run {script} --if-present',
                    target=target,  # type: ignore[arg-type]
                    source='package.json',
                    strength=strength,
                )
            )
    return commands


def _has_project_validation_marker(root: Path) -> bool:
    markers = (
        'package.json',
        'pyproject.toml',
        'pytest.ini',
        'setup.cfg',
        'tox.ini',
        'tsconfig.json',
        'Cargo.toml',
        'go.mod',
    )
    return any((root / marker).exists() for marker in markers)


def _python_commands(root: Path) -> list[ValidationCommand]:
    markers = ('pyproject.toml', 'pytest.ini', 'setup.cfg', 'tox.ini')
    if any((root / marker).is_file() for marker in markers) or (root / 'tests').is_dir():
        return [
            ValidationCommand(
                id='python:pytest',
                command='python -m pytest -q',
                target='test',
                source='python',
                strength=70,
            )
        ]
    return []


def _typescript_commands(root: Path) -> list[ValidationCommand]:
    if (root / 'tsconfig.json').is_file():
        return [
            ValidationCommand(
                id='tsc',
                command='npx tsc --noEmit',
                target='typecheck',
                source='tsconfig.json',
                strength=60,
            )
        ]
    return []


def _rust_commands(root: Path) -> list[ValidationCommand]:
    if (root / 'Cargo.toml').is_file():
        return [
            ValidationCommand(
                id='cargo:test',
                command='cargo test',
                target='test',
                source='Cargo.toml',
                strength=70,
            )
        ]
    return []


def _go_commands(root: Path) -> list[ValidationCommand]:
    if (root / 'go.mod').is_file():
        return [
            ValidationCommand(
                id='go:test',
                command='go test ./...',
                target='test',
                source='go.mod',
                strength=70,
            )
        ]
    return []
