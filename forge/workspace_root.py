'''Resolve a predictable ForgeCode workspace from CLI paths and Git metadata.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspaceLocation:
    root: Path
    cwd: Path
    source: str


def resolve_workspace(
    *,
    cwd: Path | None = None,
    root: Path | None = None,
    discover_git: bool = True,
    process_cwd: Path | None = None,
) -> WorkspaceLocation:
    '''Resolve startup cwd and workspace boundary without changing process cwd.'''
    base = (process_cwd or Path.cwd()).resolve()
    requested_cwd = resolve_directory(cwd, base=base, name='cwd') if cwd else None
    requested_root = (
        resolve_directory(root, base=base, name='root') if root else None
    )

    if requested_root is not None:
        effective_cwd = requested_cwd or requested_root
        ensure_within(effective_cwd, requested_root)
        return WorkspaceLocation(
            root=requested_root,
            cwd=effective_cwd,
            source='explicit',
        )

    effective_cwd = requested_cwd or base
    git_root = find_git_root(effective_cwd) if discover_git else None
    return WorkspaceLocation(
        root=git_root or effective_cwd,
        cwd=effective_cwd,
        source='git' if git_root is not None else 'cwd',
    )


def resolve_directory(value: Path, *, base: Path, name: str) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise ValueError(f'{name} directory does not exist: {value}')
    if not resolved.is_dir():
        raise ValueError(f'{name} is not a directory: {value}')
    return resolved


def find_git_root(start: Path) -> Path | None:
    '''Return the nearest parent containing a .git directory or worktree file.'''
    current = start.resolve()
    for candidate in (current, *current.parents):
        marker = candidate / '.git'
        if marker.is_dir() or marker.is_file():
            return candidate
    return None


def ensure_within(cwd: Path, root: Path) -> None:
    try:
        cwd.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f'cwd must be inside workspace root: {cwd} is outside {root}'
        ) from error
