'''Project validation command discovery and verification classification.'''

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePath
import re
from typing import Literal

from forge.runtime.workspace_classification import (
    ArtifactRule,
    VerificationArtifactScope,
)

ValidationTarget = Literal['auto', 'build', 'test', 'lint', 'typecheck', 'diff']
VerificationStatus = Literal[
    'passed',
    'failed',
    'timed_out',
    'invalid',
    'unavailable',
]
KNOWN_VERIFICATION_STATUSES = frozenset(
    {'passed', 'failed', 'timed_out', 'invalid', 'unavailable'}
)
REPAIR_REQUIRED_VERIFICATION_STATUSES = frozenset({'failed', 'timed_out'})


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    id: str
    command: str
    cwd: str = '.'
    target: ValidationTarget = 'auto'
    source: str = 'discovered'
    strength: int = 0


def normalize_verification_status(
    status: object,
    *,
    success: bool = False,
    timed_out: bool = False,
    exit_code: object = None,
) -> VerificationStatus:
    '''Return a known verification status with conservative fallback.'''
    normalized = str(status or '').strip().casefold()
    if normalized in KNOWN_VERIFICATION_STATUSES:
        return normalized  # type: ignore[return-value]
    if timed_out:
        return 'timed_out'
    if success:
        return 'passed'
    try:
        return 'passed' if int(exit_code) == 0 else 'failed'
    except (TypeError, ValueError):
        return 'failed'


def verification_status_requires_repair(status: object) -> bool:
    '''Return whether a verification status means this revision needs repair.'''
    return (
        normalize_verification_status(status)
        in REPAIR_REQUIRED_VERIFICATION_STATUSES
    )


def tool_result_verification_status(
    metadata: dict[str, object],
    *,
    success: bool,
) -> VerificationStatus:
    return normalize_verification_status(
        metadata.get('verification_status'),
        success=success,
        timed_out=bool(metadata.get('timed_out', False)),
        exit_code=metadata.get('exit_code'),
    )


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


def verification_artifact_scope(
    command: str,
    *,
    root: Path,
    target: ValidationTarget = 'auto',
) -> VerificationArtifactScope:
    '''Return declarative side-effect rules for a validation command.'''
    normalized = command.strip()
    lowered = normalized.casefold()
    rules: list[ArtifactRule] = []
    verification_type = _verification_type(lowered, target)
    forbidden_sources = (
        'src/**',
        'lib/**',
        'app/**',
        'packages/**/src/**',
        'test/**',
        'tests/**',
        '__tests__/**',
        '*.ts',
        '*.tsx',
        '*.js',
        '*.jsx',
        '*.py',
        '*.rs',
        '*.java',
    )

    if re.search(r'(?i)\btsc\b', normalized):
        if re.search(r'(?i)(?:--noEmit|--no-emit)\b', normalized):
            verification_type = 'typecheck'
        else:
            # tsc without --noEmit is allowed as a command, but generated
            # source-like files remain undeclared unless a build adapter below
            # claims a dedicated output directory.
            verification_type = 'typecheck'

    if re.search(r'(?i)\bvite\s+build\b', normalized):
        verification_type = 'build'
        rules.extend(_vite_output_rules(root))

    if re.search(r'(?i)\bwebpack\b|\brollup\b|\bparcel\b', normalized):
        verification_type = 'build'
        rules.extend(
            [
                ArtifactRule('dist/**', 'generated_artifact', 'bundler output'),
                ArtifactRule('build/**', 'generated_artifact', 'bundler output'),
            ]
        )

    if re.search(r'(?i)\b(?:pytest|python\s+-m\s+pytest)\b', normalized):
        verification_type = 'test'
        rules.append(ArtifactRule('.pytest_cache/**', 'cache', 'pytest cache'))

    if re.search(r'(?i)\bcoverage\b|--cov\b', normalized):
        rules.extend(
            [
                ArtifactRule('.coverage*', 'cache', 'coverage data'),
                ArtifactRule('coverage/**', 'generated_artifact', 'coverage report'),
                ArtifactRule('htmlcov/**', 'generated_artifact', 'coverage report'),
            ]
        )

    if re.search(r'(?i)\b(?:jest|vitest)\b', normalized):
        verification_type = 'test'
        rules.extend(
            [
                ArtifactRule('coverage/**', 'generated_artifact', 'test coverage'),
                ArtifactRule('.vitest/**', 'cache', 'vitest cache'),
            ]
        )

    if re.search(r'(?i)\bcargo\s+(?:test|check|clippy|build)\b', normalized):
        verification_type = 'test' if 'test' in lowered else 'build'
        rules.append(ArtifactRule('target/**', 'generated_artifact', 'cargo output'))

    if re.search(r'(?i)\b(?:gradle|gradlew)\b', normalized):
        verification_type = 'test' if 'test' in lowered else 'build'
        rules.extend(
            [
                ArtifactRule('build/**', 'generated_artifact', 'gradle output'),
                ArtifactRule('.gradle/**', 'cache', 'gradle cache'),
            ]
        )

    if re.search(r'(?i)\bmvn(?:w)?\s', normalized):
        verification_type = 'test' if 'test' in lowered else 'build'
        rules.append(ArtifactRule('target/**', 'generated_artifact', 'maven output'))

    return VerificationArtifactScope(
        verification_type=verification_type,
        read_patterns=(),
        allowed_writes=tuple(dict.fromkeys(rules)),
        forbidden_source_patterns=forbidden_sources,
        allow_network=False,
        allow_dependency_install=False,
        cleanup_generated=False,
        reusable=True,
    )


def verification_cache_key(
    *,
    source_revision: int,
    command: str,
    cwd: str,
    scope: VerificationArtifactScope,
) -> tuple[object, ...]:
    return (
        source_revision,
        command.strip(),
        cwd.replace('\\', '/'),
        scope.verification_type,
        tuple((rule.pattern, rule.kind) for rule in scope.allowed_writes),
        NON_INTERACTIVE_ENV_KEY,
    )


NON_INTERACTIVE_ENV_KEY = tuple(sorted(NON_INTERACTIVE_ENV.items()))


def _verification_type(
    lowered_command: str,
    target: ValidationTarget,
) -> Literal['auto', 'typecheck', 'build', 'test', 'lint', 'smoke']:
    if target in {'build', 'test', 'lint', 'typecheck'}:
        return target
    if 'typecheck' in lowered_command or 'type-check' in lowered_command:
        return 'typecheck'
    if 'build' in lowered_command:
        return 'build'
    if 'lint' in lowered_command:
        return 'lint'
    if 'test' in lowered_command:
        return 'test'
    return 'auto'


def _vite_output_rules(root: Path) -> list[ArtifactRule]:
    # Vite defaults to dist. Adapter-specific defaults live here, not in
    # completion, progress, or revision logic.
    rules = [ArtifactRule('dist/**', 'generated_artifact', 'vite build output')]
    config_patterns = ('vite.config.ts', 'vite.config.js', 'vite.config.mjs')
    for name in config_patterns:
        config = root / name
        if not config.is_file():
            continue
        try:
            text = config.read_text(encoding='utf-8')
        except OSError:
            continue
        for match in re.finditer(r'outDir\s*:\s*[\'"]([^\'"]+)[\'"]', text):
            raw = match.group(1).strip().replace('\\', '/')
            if raw and not raw.startswith('/') and '..' not in PurePath(raw).parts:
                rules.append(
                    ArtifactRule(
                        f'{raw.rstrip("/")}/**',
                        'generated_artifact',
                        'vite configured output',
                    )
                )
    return rules


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
            command = f'npm run {script} --if-present'
            if script == 'build':
                command = _safe_typescript_vite_build_command(root, value) or command
            commands.append(
                ValidationCommand(
                    id=f'npm:{script}',
                    command=command,
                    target=target,  # type: ignore[arg-type]
                    source='package.json',
                    strength=strength,
                )
            )
    return commands


def _safe_typescript_vite_build_command(root: Path, script: str) -> str:
    normalized = script.strip()
    if not (root / 'tsconfig.json').is_file():
        return ''
    if not re.search(r'(?i)\btsc\b', normalized):
        return ''
    if not re.search(r'(?i)\bvite\s+build\b', normalized):
        return ''
    project_match = re.search(
        r'(?i)\btsc\b[^&|;]*\s(?:-p|--project)\s+([^\s&|;]+)',
        normalized,
    )
    project = project_match.group(1) if project_match else 'tsconfig.json'
    project = project.strip('"\'')
    if Path(project).is_absolute() or '..' in Path(project).parts:
        return ''
    return f'npx tsc --noEmit -p {project} && npx vite build'


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
