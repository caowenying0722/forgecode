'''Workspace change classification shared by tracking and completion.'''

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Literal


ChangeKind = Literal[
    'source',
    'test',
    'configuration',
    'task_supporting',
    'generated_artifact',
    'cache',
    'unrelated',
    'forbidden',
    'verification_side_effect',
]
ChangeOrigin = Literal['agent', 'verification']
ArtifactKind = Literal['generated_artifact', 'cache']
ArtifactOperation = Literal['created', 'modified', 'deleted']
VerificationType = Literal['auto', 'typecheck', 'build', 'test', 'lint', 'smoke']


SOURCE_SUFFIXES = frozenset(
    {
        '.c',
        '.cc',
        '.cpp',
        '.cs',
        '.css',
        '.go',
        '.h',
        '.hpp',
        '.html',
        '.java',
        '.js',
        '.jsx',
        '.kt',
        '.kts',
        '.php',
        '.py',
        '.rb',
        '.rs',
        '.scala',
        '.scss',
        '.svelte',
        '.swift',
        '.ts',
        '.tsx',
        '.vue',
    }
)
TEST_PATH_PATTERNS = (
    'test/**',
    'tests/**',
    '__tests__/**',
    '**/*.test.*',
    '**/*.spec.*',
    '**/test_*.py',
    '**/*_test.py',
)
CONFIG_PATH_PATTERNS = (
    '.gitignore',
    'Cargo.lock',
    'Cargo.toml',
    'go.mod',
    'go.sum',
    'gradle.properties',
    'package.json',
    'package-lock.json',
    'pnpm-lock.yaml',
    'pom.xml',
    'pyproject.toml',
    'requirements*.txt',
    'rollup.config.*',
    'setup.cfg',
    'tox.ini',
    'tsconfig*.json',
    'vite.config.*',
    'webpack.config.*',
    'yarn.lock',
    'bun.lock',
    'bun.lockb',
)
SUPPORTING_PATH_PATTERNS = (
    'README*',
    'docs/**',
    'assets/**',
    'public/**',
)
TEMPORARY_PATH_PATTERNS = (
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
GENERIC_GENERATED_PATTERNS = (
    'dist/**',
    'build/**',
    'target/**',
    'coverage/**',
    'htmlcov/**',
    '**/*.map',
)
GENERIC_CACHE_PATTERNS = (
    '.cache/**',
    '.gradle/**',
    '.mypy_cache/**',
    '.pytest_cache/**',
    '.ruff_cache/**',
    '__pycache__/**',
    '**/__pycache__/**',
)
DEFAULT_FORBIDDEN_PATTERNS = (
    'tests/hidden/**',
    '**/tests/hidden/**',
)


@dataclass(frozen=True, slots=True)
class ArtifactRule:
    pattern: str
    kind: ArtifactKind = 'generated_artifact'
    description: str = ''


@dataclass(frozen=True, slots=True)
class ArtifactDelta:
    '''One generated/cache path transition produced by verification.'''

    path: str
    operation: ArtifactOperation
    kind: ArtifactKind
    before_fingerprint: str | None
    after_fingerprint: str | None
    rule_pattern: str
    rule_reason: str = ''

    def as_dict(self) -> dict[str, object]:
        return {
            'path': self.path,
            'operation': self.operation,
            'kind': self.kind,
            'before_fingerprint': self.before_fingerprint,
            'after_fingerprint': self.after_fingerprint,
            'rule_pattern': self.rule_pattern,
            'rule_reason': self.rule_reason,
        }


@dataclass(frozen=True, slots=True)
class VerificationArtifactScope:
    '''Declarative side-effect model for one verification command.'''

    verification_type: VerificationType = 'auto'
    read_patterns: tuple[str, ...] = ()
    allowed_writes: tuple[ArtifactRule, ...] = ()
    forbidden_source_patterns: tuple[str, ...] = ('src/**', 'lib/**', 'app/**')
    allow_network: bool = False
    allow_dependency_install: bool = False
    cleanup_generated: bool = False
    reusable: bool = True

    @property
    def allowed_write_patterns(self) -> tuple[str, ...]:
        return tuple(rule.pattern for rule in self.allowed_writes)


@dataclass(frozen=True, slots=True)
class ClassifiedChange:
    path: str
    kind: ChangeKind
    origin: ChangeOrigin = 'agent'
    reason: str = ''

    @property
    def affects_source_revision(self) -> bool:
        return self.kind in {
            'source',
            'test',
            'configuration',
            'task_supporting',
            'forbidden',
        }

    @property
    def task_relevant_candidate(self) -> bool:
        return self.kind in {
            'source',
            'test',
            'configuration',
            'task_supporting',
        }


@dataclass(frozen=True, slots=True)
class ChangeSetClassification:
    changes: tuple[ClassifiedChange, ...] = field(default_factory=tuple)

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.affects_source_revision
        )

    @property
    def task_candidate_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.task_relevant_candidate
        )

    @property
    def generated_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.kind == 'generated_artifact'
        )

    @property
    def cache_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path for change in self.changes if change.kind == 'cache'
        )

    @property
    def unrelated_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.kind == 'unrelated'
        )

    @property
    def forbidden_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.kind == 'forbidden'
        )

    @property
    def verification_side_effect_paths(self) -> tuple[str, ...]:
        return tuple(
            change.path
            for change in self.changes
            if change.kind == 'verification_side_effect'
        )

    @property
    def only_artifacts_or_cache(self) -> bool:
        return bool(self.changes) and all(
            change.kind in {'generated_artifact', 'cache'}
            for change in self.changes
        )


class WorkspaceChangeClassifier:
    '''Classify changed paths without embedding framework rules in callers.'''

    def classify(
        self,
        paths: tuple[str, ...],
        *,
        origin: ChangeOrigin = 'agent',
        artifact_scope: VerificationArtifactScope | None = None,
        forbidden_patterns: tuple[str, ...] = DEFAULT_FORBIDDEN_PATTERNS,
    ) -> ChangeSetClassification:
        changes = tuple(
            self.classify_path(
                path.replace('\\', '/'),
                origin=origin,
                artifact_scope=artifact_scope,
                forbidden_patterns=forbidden_patterns,
            )
            for path in paths
            if path
        )
        return ChangeSetClassification(changes=changes)

    def classify_path(
        self,
        path: str,
        *,
        origin: ChangeOrigin = 'agent',
        artifact_scope: VerificationArtifactScope | None = None,
        forbidden_patterns: tuple[str, ...] = DEFAULT_FORBIDDEN_PATTERNS,
    ) -> ClassifiedChange:
        normalized = path.replace('\\', '/')
        if _matches_any(normalized, forbidden_patterns):
            return ClassifiedChange(
                normalized,
                'forbidden',
                origin,
                'matches forbidden workspace policy',
            )

        scoped = _artifact_rule_for(normalized, artifact_scope)
        if scoped is not None:
            return ClassifiedChange(
                normalized,
                scoped.kind,
                origin,
                scoped.description or f'allowed by {scoped.pattern}',
            )

        if origin == 'verification':
            if _is_likely_handwritten(normalized, artifact_scope):
                return ClassifiedChange(
                    normalized,
                    'verification_side_effect',
                    origin,
                    'verification modified undeclared handwritten output',
                )
            return ClassifiedChange(
                normalized,
                'verification_side_effect',
                origin,
                'verification modified an undeclared path',
            )

        if _matches_any(normalized, TEMPORARY_PATH_PATTERNS):
            return ClassifiedChange(normalized, 'unrelated', origin, 'temporary path')
        if _matches_any(normalized, GENERIC_CACHE_PATTERNS):
            return ClassifiedChange(normalized, 'cache', origin, 'cache path')
        if _matches_any(normalized, GENERIC_GENERATED_PATTERNS):
            return ClassifiedChange(
                normalized,
                'generated_artifact',
                origin,
                'generated output path',
            )
        if _matches_any(normalized, TEST_PATH_PATTERNS):
            return ClassifiedChange(normalized, 'test', origin, 'test path')
        if _matches_any(normalized, CONFIG_PATH_PATTERNS):
            return ClassifiedChange(
                normalized,
                'configuration',
                origin,
                'project configuration path',
            )
        if _matches_any(normalized, SUPPORTING_PATH_PATTERNS):
            return ClassifiedChange(
                normalized,
                'task_supporting',
                origin,
                'supporting project path',
            )
        if PurePosixPath(normalized).suffix.casefold() in SOURCE_SUFFIXES:
            return ClassifiedChange(normalized, 'source', origin, 'source suffix')
        return ClassifiedChange(
            normalized,
            'task_supporting',
            origin,
            'unclassified task-local file',
        )


def _artifact_rule_for(
    path: str,
    artifact_scope: VerificationArtifactScope | None,
) -> ArtifactRule | None:
    if artifact_scope is None:
        return None
    return next(
        (rule for rule in artifact_scope.allowed_writes if _matches(path, rule.pattern)),
        None,
    )


def artifact_rule_for(
    path: str,
    artifact_scope: VerificationArtifactScope,
) -> ArtifactRule | None:
    '''Return the declared rule responsible for one artifact path.'''
    return _artifact_rule_for(path.replace('\\', '/'), artifact_scope)


def artifact_deltas_from_metadata(value: object) -> tuple[ArtifactDelta, ...]:
    '''Decode additive artifact evidence while tolerating legacy records.'''
    if not isinstance(value, (list, tuple)):
        return ()
    deltas: list[ArtifactDelta] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        operation = str(item.get('operation', ''))
        kind = str(item.get('kind', ''))
        path = str(item.get('path', ''))
        if (
            not path
            or operation not in {'created', 'modified', 'deleted'}
            or kind not in {'generated_artifact', 'cache'}
        ):
            continue
        before = item.get('before_fingerprint')
        after = item.get('after_fingerprint')
        deltas.append(
            ArtifactDelta(
                path=path,
                operation=operation,  # type: ignore[arg-type]
                kind=kind,  # type: ignore[arg-type]
                before_fingerprint=(str(before) if before is not None else None),
                after_fingerprint=(str(after) if after is not None else None),
                rule_pattern=str(item.get('rule_pattern', '')),
                rule_reason=str(item.get('rule_reason', '')),
            )
        )
    return tuple(deltas)


def _is_likely_handwritten(
    path: str,
    artifact_scope: VerificationArtifactScope | None,
) -> bool:
    if artifact_scope is not None and _matches_any(
        path,
        artifact_scope.forbidden_source_patterns,
    ):
        return True
    return (
        _matches_any(path, TEST_PATH_PATTERNS)
        or _matches_any(path, CONFIG_PATH_PATTERNS)
        or PurePosixPath(path).suffix.casefold() in SOURCE_SUFFIXES
    )


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(path, pattern) for pattern in patterns)


def _matches(path: str, pattern: str) -> bool:
    candidate = path.replace('\\', '/')
    normalized = pattern.replace('\\', '/')
    if fnmatchcase(candidate, normalized):
        return True
    return bool(
        normalized.endswith('/**')
        and candidate == normalized[:-3].rstrip('/')
    )
