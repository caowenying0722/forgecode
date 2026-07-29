'''Task-local scope inference and change relevance checks.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re

from forge.runtime.completion import matches_any
from forge.runtime.paths import normalize_workspace_path


DISPOSABLE_PATH_PATTERNS = (
    'tmp*',
    'temp*',
    'scratch*',
    '*.tmp',
    '*.temp',
    '*.bak',
    '*.orig',
    '**/tmp*',
    '**/temp*',
    '**/scratch*',
    '**/*.tmp',
    '**/*.temp',
    '**/*.bak',
    '**/*.orig',
)
PROJECT_SUPPORT_PATTERNS = (
    'package.json',
    'package-lock.json',
    'pnpm-lock.yaml',
    'yarn.lock',
    'tsconfig*.json',
    'vite.config.*',
    'webpack.config.*',
    'rollup.config.*',
    'pyproject.toml',
    'requirements*.txt',
    'Cargo.toml',
    'Cargo.lock',
    'go.mod',
    'go.sum',
    'README*',
    'docs/**',
    'tests/**',
    'test/**',
    '__tests__/**',
)
GAME_SCOPE_PATTERNS = (
    'game/**',
    'play/**',
    'src/game/**',
    'src/**/game/**',
    'src/**/scenes/**',
    'src/**/entities/**',
    'src/**/systems/**',
    'src/**/components/**',
    'src/**/physics/**',
    'src/**/assets/**',
    'public/assets/**',
    'assets/**',
    *PROJECT_SUPPORT_PATTERNS,
)
GENERIC_CODE_SCOPE_PATTERNS = (
    'src/**',
    'lib/**',
    'app/**',
    'packages/**',
    'components/**',
    'server/**',
    'client/**',
    *PROJECT_SUPPORT_PATTERNS,
)
GAME_TERMS = re.compile(
    r'(?i)(?:game|phaser|scene|player|enemy|collision|sprite|'
    r'游戏|场景|玩家|敌人|碰撞|骨架)'
)
CODE_TERMS = re.compile(
    r'(?i)(?:code|source|src|test|tests|build|lint|typecheck|'
    r'代码|源码|测试|构建)'
)
PATH_TOKEN = re.compile(
    r'(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*{}@()[\]-]*'
)
BARE_FILE_TOKEN = re.compile(
    r'(?<![\w/.-])[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+(?![\w/.-])'
)


@dataclass(frozen=True, slots=True)
class TaskScope:
    '''Deterministic approximation of what paths can satisfy a task.'''

    patterns: tuple[str, ...] = ()
    disposable_patterns: tuple[str, ...] = DISPOSABLE_PATH_PATTERNS

    @property
    def constrained(self) -> bool:
        return bool(self.patterns)


@dataclass(frozen=True, slots=True)
class ChangeRelevance:
    relevant: bool
    reasons: tuple[str, ...] = ()


def infer_task_scope(
    goal: str,
    *,
    evidence_paths: tuple[str, ...] = (),
    scope_hints: tuple[str, ...] = (),
) -> TaskScope:
    '''Infer a conservative path scope from user intent and collected evidence.'''
    patterns: list[str] = []
    patterns.extend(_patterns_from_hints(scope_hints))
    patterns.extend(_patterns_from_text(goal))
    patterns.extend(_patterns_from_evidence(evidence_paths))

    if GAME_TERMS.search(goal):
        patterns.extend(GAME_SCOPE_PATTERNS)
    elif CODE_TERMS.search(goal):
        patterns.extend(GENERIC_CODE_SCOPE_PATTERNS)

    return TaskScope(patterns=tuple(dict.fromkeys(patterns)))


def evaluate_change_relevance(
    changed_paths: tuple[str, ...],
    scope: TaskScope,
) -> ChangeRelevance:
    if not changed_paths:
        return ChangeRelevance(relevant=True)

    disposable = tuple(
        path
        for path in changed_paths
        if matches_any(path, scope.disposable_patterns)
    )
    if len(disposable) == len(changed_paths):
        return ChangeRelevance(
            relevant=False,
            reasons=(
                'The only workspace changes are disposable or temporary '
                'files: ' + ', '.join(disposable),
            ),
        )

    if not scope.constrained:
        return ChangeRelevance(relevant=True)

    relevant_paths = tuple(
        path for path in changed_paths if matches_any(path, scope.patterns)
    )
    if relevant_paths:
        return ChangeRelevance(relevant=True)

    return ChangeRelevance(
        relevant=False,
        reasons=(
            'The workspace changed, but no changed path matches the inferred '
            'task scope. Changed paths: '
            + ', '.join(changed_paths)
            + '. Inferred scope: '
            + ', '.join(scope.patterns[:12]),
        ),
    )


def render_task_scope_context(scope: TaskScope) -> str:
    if not scope.constrained:
        return ''
    patterns = ', '.join(scope.patterns[:16])
    suffix = ' ...' if len(scope.patterns) > 16 else ''
    return (
        '[ForgeCode Task Scope]\n'
        'Completion requires task-relevant workspace changes, not merely any '
        'Diff. Temporary placeholder files do not count as progress. '
        f'Current inferred relevant path patterns: {patterns}{suffix}.'
    )


def _patterns_from_hints(values: tuple[str, ...]) -> list[str]:
    patterns: list[str] = []
    for value in values:
        patterns.extend(_expand_path_pattern(value))
    return patterns


def _patterns_from_text(text: str) -> list[str]:
    patterns: list[str] = []
    for match in PATH_TOKEN.finditer(text):
        patterns.extend(_expand_path_pattern(match.group(0)))
    for match in BARE_FILE_TOKEN.finditer(text):
        patterns.extend(_expand_path_pattern(match.group(0)))
    return patterns


def _patterns_from_evidence(paths: tuple[str, ...]) -> list[str]:
    patterns: list[str] = []
    for path in paths:
        normalized = normalize_workspace_path(path)
        if not normalized or normalized == '.':
            continue
        patterns.append(normalized)
        parent = PurePosixPath(normalized).parent.as_posix()
        if parent and parent != '.':
            patterns.append(f'{parent}/**')
    return patterns


def _expand_path_pattern(raw_value: str) -> list[str]:
    value = raw_value.strip().strip('`"\'').replace('\\', '/')
    if not value:
        return []
    normalized = normalize_workspace_path(value)
    if normalized == '.':
        return []
    if any(token in normalized for token in '*?['):
        return [normalized]
    if normalized.endswith('/'):
        normalized = normalized.rstrip('/')
        return [f'{normalized}/**']
    suffix = PurePosixPath(normalized).suffix
    if suffix:
        return [normalized]
    return [normalized, f'{normalized}/**']
