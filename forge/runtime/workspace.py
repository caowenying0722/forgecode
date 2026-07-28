'''Git-backed working tree tracking for completion evidence.'''

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath

from forge.runtime.paths import normalize_workspace_path
from forge.runtime.process import run_process

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


@dataclass(frozen=True, slots=True)
class WorkspaceChange:
    '''One observed transition between working tree revisions.'''

    revision: int
    paths: tuple[str, ...]


class WorkspaceTracker:
    '''Track task-local changes without treating prior user edits as Agent work.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.baseline = WorkspaceSnapshot()
        self.current = WorkspaceSnapshot()
        self.revision = 0
        self.available = False
        self._watched_paths: set[str] = set()

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

    async def refresh(self) -> WorkspaceChange | None:
        '''Capture tool-caused changes and advance the revision when needed.'''
        snapshot = await self._capture()
        if snapshot is None:
            self.available = False
            return None
        self.available = True
        paths = changed_paths(self.current, snapshot)
        if not paths:
            return None
        self.current = snapshot
        self.revision += 1
        return WorkspaceChange(revision=self.revision, paths=paths)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        '''Return only paths whose content differs from the turn baseline.'''
        return changed_paths(self.baseline, self.current)

    async def _capture(self) -> WorkspaceSnapshot | None:
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
