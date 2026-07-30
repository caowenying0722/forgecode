'''Task-local scope inference and change relevance checks.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Literal

from forge.runtime.completion import matches_any
from forge.runtime.paths import normalize_workspace_path
from forge.runtime.workspace_classification import WorkspaceChangeClassifier


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


ChangePathRole = Literal[
    'required',
    'supporting',
    'generated',
    'unrelated',
    'forbidden',
]


@dataclass(frozen=True, slots=True)
class ClassifiedPath:
    path: str
    role: ChangePathRole


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

    classified = classify_changed_paths(changed_paths, scope)
    blocked = tuple(
        item for item in classified if item.role in {'forbidden', 'unrelated'}
    )
    if classified and len(blocked) == len(classified):
        return ChangeRelevance(
            relevant=False,
            reasons=(
                'The only workspace changes are unrelated, forbidden, '
                'temporary files: '
                + ', '.join(item.path for item in blocked),
            ),
        )
    if blocked:
        return ChangeRelevance(
            relevant=False,
            reasons=(
                'The workspace changed, but these paths are outside the '
                'task scope or are forbidden: '
                + ', '.join(item.path for item in blocked),
            ),
        )
    if not scope.constrained:
        return ChangeRelevance(relevant=True)

    relevant_paths = tuple(
        item.path for item in classified if item.role in {'required', 'supporting'}
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


def classify_changed_paths(
    changed_paths: tuple[str, ...],
    scope: TaskScope,
) -> tuple[ClassifiedPath, ...]:
    normalized = tuple(
        path.replace('\\', '/') for path in changed_paths if path
    )
    result: list[ClassifiedPath] = []
    workspace_classifier = WorkspaceChangeClassifier()
    for path in normalized:
        base = workspace_classifier.classify_path(path)
        if base.kind in {'generated_artifact', 'cache', 'unrelated'}:
            result.append(ClassifiedPath(path, 'unrelated'))
        elif scope.constrained and _matches_scope(path, scope.patterns):
            role: ChangePathRole = (
                'supporting'
                if _matches_scope(path, PROJECT_SUPPORT_PATTERNS)
                else 'required'
            )
            result.append(ClassifiedPath(path, role))
        elif not scope.constrained:
            result.append(ClassifiedPath(path, 'required'))
        else:
            result.append(ClassifiedPath(path, 'unrelated'))
    return tuple(result)


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


def _matches_scope(path: str, patterns: tuple[str, ...]) -> bool:
    if matches_any(path, patterns):
        return True
    return any(
        pattern.endswith('/**') and path == pattern[:-3].rstrip('/')
        for pattern in patterns
    )
