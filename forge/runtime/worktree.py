'''Git worktree isolation and optimistic integration for subagents.'''

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable
from uuid import uuid4

from forge.runtime.workspace import fingerprint_path, parse_porcelain_paths


@dataclass(frozen=True, slots=True)
class WorktreeLease:
    id: str
    agent: str
    repository: Path
    path: Path
    base_head: str
    baseline: dict[str, str]


@dataclass(frozen=True, slots=True)
class WorktreeIntegration:
    changed_paths: tuple[str, ...]
    integrated_paths: tuple[str, ...]
    conflicts: tuple[str, ...]
    worktree_path: str | None
    cleaned_up: bool

    @property
    def success(self) -> bool:
        return not self.conflicts


class WorktreeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SubagentWorktreeManager:
    '''Create isolated subagent checkouts and merge without overwrites.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'worktrees'

    def create(self, agent: str = 'subagent') -> WorktreeLease:
        repository = self._repository_root()
        if repository != self.root:
            raise WorktreeError(
                'subagent_worktree_root_mismatch',
                'Worktree-isolated subagents currently require ForgeCode to '
                f'run at the Git repository root ({repository}).',
            )
        dirty_paths = self._dirty_paths(repository)
        head = self._git(repository, 'rev-parse', 'HEAD').stdout.strip()
        if not head:
            raise WorktreeError(
                'subagent_worktree_no_head',
                'The repository has no Git commit to use as a worktree base.',
            )
        lease_id = f'{clean_name(agent)}-{uuid4().hex[:12]}'
        path = self.directory / lease_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git(
            repository,
            'worktree',
            'add',
            '--detach',
            str(path),
            head,
        )
        try:
            self._sync_dirty_state(repository, path, dirty_paths)
            baseline = self._snapshot(path)
        except Exception:
            self._remove(path)
            raise
        return WorktreeLease(
            id=lease_id,
            agent=clean_name(agent),
            repository=repository,
            path=path.resolve(),
            base_head=head,
            baseline=baseline,
        )

    def describe(self) -> str:
        if not self.directory.is_dir():
            return 'No preserved subagent worktrees.'
        lines: list[str] = []
        for path in sorted(self.directory.iterdir()):
            if not path.is_dir():
                continue
            result = subprocess.run(
                ['git', 'status', '--short', '--branch'],
                cwd=path,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            status = result.stdout.strip() or 'clean'
            lines.append(f'- {path.name}\n  path: {path}\n  {status}')
        return '\n'.join(lines) if lines else 'No preserved subagent worktrees.'

    def integrate(
        self,
        lease: WorktreeLease,
        *,
        apply: bool = True,
    ) -> WorktreeIntegration:
        self._validate_lease(lease)
        final = self._snapshot(lease.path)
        changed = changed_paths(lease.baseline, final)
        if not changed:
            cleaned = self._remove(lease.path)
            return WorktreeIntegration((), (), (), None, cleaned)
        if not apply:
            return WorktreeIntegration(
                changed,
                (),
                (),
                str(lease.path),
                False,
            )

        with integration_lock(lease.repository):
            conflicts = tuple(
                path
                for path in changed
                if not self._can_integrate(path, lease.baseline, final)
            )
            if conflicts:
                return WorktreeIntegration(
                    changed,
                    (),
                    conflicts,
                    str(lease.path),
                    False,
                )

            integrated: list[str] = []
            for relative in changed:
                source = lease.path / Path(relative)
                target = lease.repository / Path(relative)
                final_fingerprint = final.get(relative, 'missing')
                if (
                    fingerprint_path(lease.repository, relative)
                    == final_fingerprint
                ):
                    continue
                copy_path_state(source, target, final_fingerprint)
                integrated.append(relative)
        cleaned = self._remove(lease.path)
        return WorktreeIntegration(
            changed,
            tuple(integrated),
            (),
            None if cleaned else str(lease.path),
            cleaned,
        )

    def _can_integrate(
        self,
        relative: str,
        baseline: dict[str, str],
        final: dict[str, str],
    ) -> bool:
        current = fingerprint_path(self.root, relative)
        return current in {
            baseline.get(relative, 'missing'),
            final.get(relative, 'missing'),
        }

    def _repository_root(self) -> Path:
        try:
            result = self._git(self.root, 'rev-parse', '--show-toplevel')
        except WorktreeError as error:
            raise WorktreeError(
                'subagent_worktree_unavailable',
                'Writing subagents require a Git repository with at least '
                'one commit; no shared-directory fallback was used.',
            ) from error
        return Path(result.stdout.strip()).resolve()

    def _dirty_paths(self, repository: Path) -> tuple[str, ...]:
        result = self._git(
            repository,
            'status',
            '--porcelain=v1',
            '-z',
            '--untracked-files=all',
            '--ignored=no',
        )
        return tuple(
            path
            for path in parse_porcelain_paths(result.stdout)
            if included_path(path)
        )

    def _sync_dirty_state(
        self,
        repository: Path,
        worktree: Path,
        paths: Iterable[str],
    ) -> None:
        for relative in paths:
            source = repository / Path(relative)
            target = worktree / Path(relative)
            state = fingerprint_path(repository, relative)
            copy_path_state(source, target, state)

    def _snapshot(self, root: Path) -> dict[str, str]:
        result = self._git(
            root,
            'ls-files',
            '-z',
            '--cached',
            '--others',
            '--exclude-standard',
        )
        paths = {
            path.replace('\\', '/')
            for path in result.stdout.split('\0')
            if path and included_path(path.replace('\\', '/'))
        }
        return {
            path: fingerprint_path(root, path)
            for path in sorted(paths)
        }

    def _validate_lease(self, lease: WorktreeLease) -> None:
        path = lease.path.resolve()
        expected_parent = self.directory.resolve()
        if path.parent != expected_parent or not path.is_dir():
            raise WorktreeError(
                'invalid_subagent_worktree',
                f'Invalid or missing subagent worktree: {path}',
            )

    def _remove(self, path: Path) -> bool:
        resolved = path.resolve()
        if resolved.parent != self.directory.resolve():
            raise WorktreeError(
                'invalid_subagent_worktree',
                f'Refusing to remove worktree outside {self.directory}.',
            )
        result = subprocess.run(
            ['git', 'worktree', 'remove', '--force', str(resolved)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode == 0:
            return True
        return not resolved.exists()

    @staticmethod
    def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ['git', *arguments],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise WorktreeError(
                'subagent_worktree_git_failed',
                f'Git worktree command failed: {error}',
            ) from error
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout).strip()
            raise WorktreeError(
                'subagent_worktree_git_failed',
                diagnostic or f'git {arguments[0]} failed.',
            )
        return result


def changed_paths(
    baseline: dict[str, str],
    final: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        path
        for path in sorted(set(baseline) | set(final))
        if baseline.get(path, 'missing') != final.get(path, 'missing')
    )


def copy_path_state(source: Path, target: Path, fingerprint: str) -> None:
    if fingerprint == 'missing':
        if target.is_file() or target.is_symlink():
            target.unlink()
        return
    if fingerprint == 'directory':
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            raise WorktreeError(
                'subagent_worktree_path_conflict',
                f'Cannot replace directory with file: {target}',
            )
        target.unlink()
    if source.is_symlink():
        target.symlink_to(source.readlink(), target_is_directory=source.is_dir())
    else:
        shutil.copy2(source, target)


def included_path(path: str) -> bool:
    normalized = path.replace('\\', '/').lstrip('./')
    return bool(normalized) and not (
        normalized == '.forge' or normalized.startswith('.forge/')
    )


def clean_name(value: str) -> str:
    cleaned = ''.join(
        character if character.isalnum() or character in {'-', '_'} else '-'
        for character in value.casefold()
    ).strip('-_')
    return cleaned[:40] or 'subagent'


@contextmanager
def integration_lock(repository: Path):
    '''Serialize optimistic checks and writes across ForgeCode agents.'''
    result = subprocess.run(
        ['git', 'rev-parse', '--git-common-dir'],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise WorktreeError(
            'subagent_integration_lock_failed',
            'Could not resolve the shared Git directory for integration.',
        )
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = (repository / common).resolve()
    lock_path = common / 'forgecode-worktree.lock'
    deadline = time.monotonic() + 30
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(descriptor, str(os.getpid()).encode('ascii'))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120
            except OSError:
                stale = False
            if stale:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise WorktreeError(
                    'subagent_integration_lock_timeout',
                    'Timed out waiting for another subagent integration.',
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass
