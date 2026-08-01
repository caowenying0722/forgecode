'''Git-backed working tree tracking for completion evidence.'''

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath

from forge.runtime.paths import normalize_workspace_path
from forge.runtime.process import run_process
from forge.runtime.workspace_classification import (
    ChangeSetClassification,
    VerificationArtifactScope,
    WorkspaceChangeClassifier,
)

DEFAULT_UNWATCHED_PARTS = frozenset(
    {
        '.cache',
        '.conda',
        '.conda-pkgs',
        '.forge',
        '.git',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        '.venv',
        '__pycache__',
        'node_modules',
    }
)
INTERNAL_GIT_DIRECTORY = Path('.forge') / 'git'
INTERNAL_GIT_EXCLUDE = '.forge/'


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    '''Content state for paths currently changed relative to Git HEAD.'''

    files: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        '''Return a stable identity for this observed workspace state.'''
        digest = sha256()
        for path, fingerprint in sorted(self.files.items()):
            digest.update(path.encode('utf-8'))
            digest.update(b'\0')
            digest.update(fingerprint.encode('utf-8'))
            digest.update(b'\0')
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WorkspaceChange:
    '''One observed transition between working tree revisions.'''

    revision: int
    paths: tuple[str, ...]
    filesystem_revision: int = 0
    source_revision: int = 0
    source_paths: tuple[str, ...] = ()
    classification: ChangeSetClassification = field(
        default_factory=ChangeSetClassification
    )
    created_paths: tuple[str, ...] = ()
    modified_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    before_snapshot_id: str = ''
    after_snapshot_id: str = ''
    before_fingerprints: tuple[tuple[str, str], ...] = ()
    after_fingerprints: tuple[tuple[str, str], ...] = ()


class WorkspaceTracker:
    '''Track task-local changes without treating prior user edits as Agent work.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.baseline = WorkspaceSnapshot()
        self.current = WorkspaceSnapshot()
        self.revision = 0
        self.filesystem_revision = 0
        self.source_revision = 0
        self.available = False
        self._watched_paths: set[str] = set()
        self._artifact_paths: set[str] = set()
        self.classifier = WorkspaceChangeClassifier()
        self.last_classification = ChangeSetClassification()
        self.verification_cache: dict[tuple[object, ...], object] = {}

    async def begin_turn(self) -> None:
        '''Use the current working tree as the immutable baseline for one turn.'''
        self._watched_paths.clear()
        snapshot = await self._capture()
        if snapshot is None and not (self.root / '.git').exists():
            await self._initialize_internal_repository()
            snapshot = await self._capture()
        self.available = snapshot is not None
        resolved = snapshot or WorkspaceSnapshot()
        self.baseline = resolved
        self.current = resolved
        self.revision = 0
        self.filesystem_revision = 0
        self.source_revision = 0
        self._artifact_paths.clear()
        self.last_classification = ChangeSetClassification()

    async def _initialize_internal_repository(self) -> bool:
        '''Create a private gitdir pointer for an otherwise non-Git workspace.'''
        control_dir = self.root / INTERNAL_GIT_DIRECTORY.parent
        if control_dir.exists() and (
            control_dir.is_symlink()
            or (
                hasattr(control_dir, 'is_junction')
                and control_dir.is_junction()
            )
        ):
            return False
        try:
            control_dir.mkdir(parents=True, exist_ok=True)
            git_dir = (self.root / INTERNAL_GIT_DIRECTORY).resolve()
            git_dir.relative_to(self.root)
        except OSError:
            return False
        except ValueError:
            return False

        result = await run_process(
            [
                'git',
                'init',
                '--quiet',
                f'--separate-git-dir={git_dir}',
                '.',
            ],
            cwd=self.root,
            timeout_seconds=30,
        )
        if result.exit_code != 0:
            return False

        exclude_path = git_dir / 'info' / 'exclude'
        try:
            existing = exclude_path.read_text(encoding='utf-8')
            patterns = {
                line.strip()
                for line in existing.splitlines()
                if line.strip() and not line.lstrip().startswith('#')
            }
            if INTERNAL_GIT_EXCLUDE not in patterns:
                separator = '' if not existing or existing.endswith('\n') else '\n'
                exclude_path.write_text(
                    f'{existing}{separator}{INTERNAL_GIT_EXCLUDE}\n',
                    encoding='utf-8',
                )
        except OSError:
            return False

        baseline = await run_process(
            [
                'git',
                '-c',
                'user.name=ForgeCode',
                '-c',
                'user.email=forgecode@local',
                '-c',
                'commit.gpgsign=false',
                'commit',
                '--quiet',
                '--allow-empty',
                '-m',
                'ForgeCode workspace baseline',
            ],
            cwd=self.root,
            timeout_seconds=30,
        )
        return baseline.exit_code == 0

    def watch_paths(self, paths: tuple[str, ...]) -> None:
        '''Capture task baselines for tool targets, including ignored files.'''
        for raw_path in paths:
            candidate = Path(raw_path)
            if candidate.is_absolute():
                continue
            resolved = (self.root / candidate).resolve(strict=False)
            try:
                relative = resolved.relative_to(self.root)
            except ValueError:
                continue
            normalized = normalize_workspace_path(str(relative))
            if normalized in self._watched_paths:
                continue
            fingerprint = fingerprint_path(self.root, normalized)
            self._watched_paths.add(normalized)
            self.baseline = WorkspaceSnapshot(
                files={**self.baseline.files, normalized: fingerprint}
            )
            self.current = WorkspaceSnapshot(
                files={**self.current.files, normalized: fingerprint}
            )

    async def refresh(
        self,
        *,
        origin: str = 'agent',
        artifact_scope: VerificationArtifactScope | None = None,
        before_snapshot: WorkspaceSnapshot | None = None,
    ) -> WorkspaceChange | None:
        '''Capture tool-caused changes and advance the revision when needed.'''
        snapshot = await self._capture(artifact_scope=artifact_scope)
        if snapshot is None:
            self.available = False
            return None
        self.available = True
        before = self.current
        transaction_before = before_snapshot or before
        paths = changed_paths(transaction_before, snapshot)
        state_snapshot = snapshot
        if before_snapshot is not None:
            state_files = dict(snapshot.files)
            for path, fingerprint in before_snapshot.files.items():
                if (
                    path not in before.files
                    and snapshot.files.get(path) == fingerprint
                ):
                    state_files.pop(path, None)
            state_snapshot = WorkspaceSnapshot(files=state_files)
        classification = self.classifier.classify(
            paths,
            origin='verification' if origin == 'verification' else 'agent',
            artifact_scope=artifact_scope,
        )
        self.current = state_snapshot
        if paths:
            self.filesystem_revision += 1
        self.revision = self.filesystem_revision
        source_paths = tuple(
            dict.fromkeys(
                (
                    *classification.source_paths,
                    *(
                        path
                        for path in paths
                        if path in self._watched_paths and origin != 'verification'
                    ),
                    *(
                        path
                        for path in classification.verification_side_effect_paths
                        if PurePosixPath(path).suffix.casefold()
                        in {'.ts', '.tsx', '.py', '.rs', '.java'}
                    ),
                )
            )
        )
        if paths and source_paths:
            self.source_revision += 1
        self.last_classification = classification
        self._artifact_paths.update(classification.generated_paths)
        self._artifact_paths.update(classification.cache_paths)
        return WorkspaceChange(
            revision=self.filesystem_revision,
            paths=paths,
            filesystem_revision=self.filesystem_revision,
            source_revision=self.source_revision,
            source_paths=source_paths,
            classification=classification,
            created_paths=tuple(
                path for path in paths
                if transaction_before.files.get(path, 'missing') == 'missing'
                and snapshot.files.get(path, 'missing') != 'missing'
            ),
            modified_paths=tuple(
                path for path in paths
                if transaction_before.files.get(path, 'missing') != 'missing'
                and snapshot.files.get(path, 'missing') != 'missing'
            ),
            deleted_paths=tuple(
                path for path in paths
                if transaction_before.files.get(path, 'missing') != 'missing'
                and snapshot.files.get(path, 'missing') == 'missing'
            ),
            before_snapshot_id=transaction_before.id,
            after_snapshot_id=snapshot.id,
            before_fingerprints=tuple(
                (path, transaction_before.files.get(path, 'missing'))
                for path in paths
            ),
            after_fingerprints=tuple(
                (path, snapshot.files.get(path, 'missing'))
                for path in paths
            ),
        )

    async def capture_transaction_snapshot(
        self,
        artifact_scope: VerificationArtifactScope,
    ) -> WorkspaceSnapshot | None:
        '''Capture declared paths immediately before a verification command.'''
        return await self._capture(artifact_scope=artifact_scope)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        '''Return task-relevant source/test/config/supporting changed paths.'''
        return self.source_changed_paths

    @property
    def source_changed_paths(self) -> tuple[str, ...]:
        '''Return changed paths that affect the source revision.'''
        all_paths = changed_paths(self.baseline, self.current)
        classification = self.classifier.classify(all_paths)
        non_source_side_effects = {
            path
            for path in self.last_classification.verification_side_effect_paths
            if PurePosixPath(path).suffix.casefold() not in {'.ts', '.tsx', '.py', '.rs', '.java'}
        }
        return tuple(
            path
            for path in dict.fromkeys(
                (
                    *classification.source_paths,
                    *(
                        path
                        for path in all_paths
                        if path in self._watched_paths
                    ),
                )
            )
            if path not in non_source_side_effects
            and not (path in self._artifact_paths and path not in self._watched_paths)
        )

    @property
    def filesystem_changed_paths(self) -> tuple[str, ...]:
        '''Return all observed paths whose content differs from turn baseline.'''
        return changed_paths(self.baseline, self.current)

    async def _capture(
        self,
        *,
        artifact_scope: VerificationArtifactScope | None = None,
    ) -> WorkspaceSnapshot | None:
        result = await run_process(
            [
                'git',
                'status',
                '--porcelain=v1',
                '-z',
                '--untracked-files=all',
                '--ignored=no',
            ],
            cwd=self.root,
            timeout_seconds=30,
        )
        if result.exit_code != 0:
            return None

        files = {
            path: fingerprint_path(self.root, path)
            for path in parse_porcelain_paths(result.stdout)
            if not should_skip_workspace_path(path)
        }
        for path in self._watched_paths:
            files[path] = fingerprint_path(self.root, path)
        for path in self._artifact_paths:
            fingerprint = fingerprint_path(self.root, path)
            if fingerprint != 'missing' or path in self.baseline.files:
                files[path] = fingerprint
        if artifact_scope is not None:
            for path in scan_artifact_scope_paths(self.root, artifact_scope):
                files[path] = fingerprint_path(self.root, path)
        return WorkspaceSnapshot(files=files)


def changed_paths(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> tuple[str, ...]:
    '''Return deterministic paths whose content state differs.'''
    paths = set(before.files) | set(after.files)
    return tuple(
        sorted(
            path
            for path in paths
            if before.files.get(path) != after.files.get(path)
        )
    )


def parse_porcelain_paths(output: str) -> tuple[str, ...]:
    '''Extract paths from ``git status --porcelain=v1 -z`` output.'''
    records = output.split('\0')
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record or len(record) < 4:
            continue
        status = record[:2]
        paths.append(normalize_workspace_path(record[3:]))
        if 'R' in status or 'C' in status:
            if index < len(records) and records[index]:
                paths.append(normalize_workspace_path(records[index]))
                index += 1
    return tuple(dict.fromkeys(paths))


def should_skip_workspace_path(path: str) -> bool:
    '''Ignore local dependency and tool caches during broad Git snapshots.'''
    return any(part in DEFAULT_UNWATCHED_PARTS for part in PurePosixPath(path).parts)


def scan_artifact_scope_paths(
    root: Path,
    artifact_scope: VerificationArtifactScope,
) -> tuple[str, ...]:
    '''Find files covered by declared verification artifact/cache patterns.'''
    paths: list[str] = []
    for pattern in artifact_scope.allowed_write_patterns:
        paths.extend(_scan_pattern(root, pattern))
    return tuple(dict.fromkeys(sorted(paths)))


def _scan_pattern(root: Path, pattern: str) -> list[str]:
    normalized = pattern.replace('\\', '/')
    parts = PurePosixPath(normalized).parts
    prefix_parts: list[str] = []
    for part in parts:
        if any(token in part for token in '*?['):
            break
        prefix_parts.append(part)
    base = root.joinpath(*prefix_parts) if prefix_parts else root
    if not base.exists():
        return []
    candidates = [base] if base.is_file() else list(base.rglob('*'))
    result: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            relative = candidate.resolve().relative_to(root)
        except ValueError:
            continue
        path = normalize_workspace_path(str(relative))
        if _matches_artifact_pattern(path, normalized):
            result.append(path)
    return result


def _matches_artifact_pattern(path: str, pattern: str) -> bool:
    from fnmatch import fnmatchcase

    candidate = path.replace('\\', '/')
    if fnmatchcase(candidate, pattern):
        return True
    return bool(pattern.endswith('/**') and candidate == pattern[:-3].rstrip('/'))


def fingerprint_path(root: Path, relative_path: str) -> str:
    '''Hash file content without following a repository symlink.'''
    path = root / Path(relative_path)
    try:
        if path.is_symlink():
            return f'symlink:{os.readlink(path)}'
        if not path.exists():
            return 'missing'
        if path.is_dir():
            return 'directory'
    except OSError as error:
        return f'unreadable:{type(error).__name__}:{error.errno}'
    digest = sha256()
    try:
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as error:
        return f'unreadable:{type(error).__name__}:{error.errno}'
    return f'file:{digest.hexdigest()}'
