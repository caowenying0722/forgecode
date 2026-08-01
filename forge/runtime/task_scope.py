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
    'index.html',
    'public/**',
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
PACKAGE_LOCKFILE_PATHS = frozenset(
    {'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock'}
)
PATH_TOKEN = re.compile(
    r'(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*{}@()[\]-]*'
)
BARE_FILE_TOKEN = re.compile(
    r'(?<![\w/.-])[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+(?![\w/.-])'
)
SCOPE_HINT_PATH_CONTEXT = re.compile(
    r'(?i)(?:scope|focus|path|paths|file|files|under|within|touch|modify|'
    r'change|edit|主要修改|修改|改动|编辑|路径|范围|文件|目录包括|涉及)'
)
EXPLICIT_PATH_HINT = re.compile(
    r'@?(?:'
    r'(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*{}@()[\]-]*|'
    r'[A-Za-z0-9_.-]+/|'
    r'[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+|'
    r'[A-Za-z0-9_.-]*[*?][A-Za-z0-9_.*{}@()[\]-]*'
    r')'
)
TASK_DOCUMENT_CONTEXT_CHANGE = re.compile(
    r'(?i)(?:read|inspect|review|understand|according to|based on|'
    r'阅读|读取|查看|明确|理解|分析|按照|按|根据|严格按照)'
    r'.{0,80}(?:task\.md|requirements?\.md|任务文档|需求文档|任务|需求|说明)'
    r'.{0,100}(?:implement|build|create|write|complete|develop|'
    r'实现|落地|创建|编写|完成|开发|开始|补齐)'
    r'|(?:task\.md|requirements?\.md|任务文档|需求文档|任务|需求|说明)'
    r'.{0,100}(?:implement|build|create|write|complete|develop|'
    r'实现|落地|创建|编写|完成|开发|开始|补齐)'
    r'|(?:implement|build|create|write|complete|develop|'
    r'实现|落地|创建|编写|完成|开发|开始|补齐)'
    r'.{0,100}(?:task\.md|requirements?\.md|任务文档|需求文档|任务|需求|说明)'
)
TASK_DOCUMENT_EDIT = re.compile(
    r'(?i)(?:edit|modify|update|rewrite|fix|'
    r'编辑|修改|更新|改写|修复)'
    r'.{0,40}(?:task\.md|requirements?\.md|任务文档|需求文档|说明文档)'
)


PatternSource = Literal[
    'goal',
    'scope_hint',
    'allowed_path',
    'evidence',
]


@dataclass(frozen=True, slots=True)
class TaskScopePattern:
    pattern: str
    source: PatternSource


@dataclass(frozen=True, slots=True)
class TaskScope:
    '''Deterministic approximation of what paths can satisfy a task.'''

    patterns: tuple[str, ...] = ()
    pattern_sources: tuple[TaskScopePattern, ...] = ()
    disposable_patterns: tuple[str, ...] = DISPOSABLE_PATH_PATTERNS

    @property
    def constrained(self) -> bool:
        return bool(self.patterns)

    @property
    def source_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source for item in self.pattern_sources))


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
    scope_hint_source: Literal['scope_hint', 'allowed_path'] = 'scope_hint',
) -> TaskScope:
    '''Infer a conservative path scope from user intent and collected evidence.'''
    entries: list[TaskScopePattern] = []
    entries.extend(
        _pattern_entries(
            scope_patterns_from_hints(scope_hints),
            source=scope_hint_source,
        )
    )
    entries.extend(_pattern_entries(_patterns_from_goal(goal), source='goal'))
    evidence_scope_paths = evidence_paths
    if (
        TASK_DOCUMENT_CONTEXT_CHANGE.search(goal)
        and not TASK_DOCUMENT_EDIT.search(goal)
    ):
        evidence_scope_paths = tuple(
            path
            for path in evidence_paths
            if PurePosixPath(path).suffix.casefold() not in {'.md', '.markdown'}
        )
    entries.extend(
        _pattern_entries(
            _patterns_from_evidence(evidence_scope_paths),
            source='evidence',
        )
    )

    return _dedupe_scope(entries)


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
    normalized_set = set(normalized)
    result: list[ClassifiedPath] = []
    workspace_classifier = WorkspaceChangeClassifier()
    for path in normalized:
        base = workspace_classifier.classify_path(path)
        if base.kind in {'generated_artifact', 'cache', 'unrelated'}:
            result.append(ClassifiedPath(path, 'unrelated'))
        elif (
            path in PACKAGE_LOCKFILE_PATHS
            and 'package.json' in normalized_set
        ):
            result.append(ClassifiedPath(path, 'supporting'))
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


def scope_patterns_from_hints(values: tuple[str, ...]) -> tuple[str, ...]:
    '''Extract only explicit workspace-relative path/glob hints.

    Free-form prose is context, not a path constraint. Mixed prose is accepted
    only when it clearly frames embedded paths as the intended scope.
    '''
    patterns: list[str] = []
    for value in values:
        patterns.extend(_patterns_from_scope_hint(value))
    return tuple(dict.fromkeys(patterns))


def _patterns_from_hints(values: tuple[str, ...]) -> list[str]:
    return list(scope_patterns_from_hints(values))


def _patterns_from_scope_hint(value: str) -> list[str]:
    stripped = value.strip()
    if _is_explicit_path_hint(stripped):
        return _expand_path_pattern(stripped)
    if not SCOPE_HINT_PATH_CONTEXT.search(stripped):
        return []
    return _patterns_from_text(stripped)


def _patterns_from_goal(text: str) -> list[str]:
    patterns = _patterns_from_text(text)
    if (
        TASK_DOCUMENT_CONTEXT_CHANGE.search(text)
        and not TASK_DOCUMENT_EDIT.search(text)
    ):
        patterns = [
            pattern
            for pattern in patterns
            if PurePosixPath(pattern).suffix.casefold() not in {'.md', '.markdown'}
        ]
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
    value = raw_value.strip().strip('`"\'').removeprefix('@').replace('\\', '/')
    if not value:
        return []
    if not _is_safe_relative_pattern(value):
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


def _is_explicit_path_hint(value: str) -> bool:
    candidate = value.strip().strip('`"\'').replace('\\', '/')
    if not candidate or any(char.isspace() for char in candidate):
        return False
    if not EXPLICIT_PATH_HINT.fullmatch(candidate):
        return False
    return _is_safe_relative_pattern(candidate.removeprefix('@'))


def _is_safe_relative_pattern(value: str) -> bool:
    normalized = value.replace('\\', '/')
    if normalized.startswith('/') or re.match(r'^[A-Za-z]:', normalized):
        return False
    parts = [part for part in normalized.split('/') if part]
    return '..' not in parts


def _pattern_entries(
    patterns: tuple[str, ...] | list[str],
    *,
    source: PatternSource,
) -> list[TaskScopePattern]:
    return [TaskScopePattern(pattern, source) for pattern in patterns]


def _dedupe_scope(entries: list[TaskScopePattern]) -> TaskScope:
    deduped: list[TaskScopePattern] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.pattern in seen:
            continue
        seen.add(entry.pattern)
        deduped.append(entry)
    return TaskScope(
        patterns=tuple(entry.pattern for entry in deduped),
        pattern_sources=tuple(deduped),
    )


def _matches_scope(path: str, patterns: tuple[str, ...]) -> bool:
    if matches_any(path, patterns):
        return True
    return any(
        pattern.endswith('/**') and path == pattern[:-3].rstrip('/')
        for pattern in patterns
    )
