'''Tests for isolated subagent Git worktrees.'''

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import pytest

from forge.runtime.worktree import SubagentWorktreeManager, WorktreeError


def test_worktrees_isolate_agents_and_reject_conflicting_merge(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manager = SubagentWorktreeManager(repository)
    first = manager.create('agent-a')
    second = manager.create('agent-b')

    (first.path / 'shared.txt').write_text('agent a\n', encoding='utf-8')
    (second.path / 'shared.txt').write_text('agent b\n', encoding='utf-8')

    first_result = manager.integrate(first)
    second_result = manager.integrate(second)

    assert first_result.success is True
    assert first_result.integrated_paths == ('shared.txt',)
    assert not first.path.exists()
    assert second_result.success is False
    assert second_result.conflicts == ('shared.txt',)
    assert second_result.worktree_path == str(second.path)
    assert second.path.exists()
    assert (repository / 'shared.txt').read_text(encoding='utf-8') == 'agent a\n'
    assert second.id in manager.describe()


def test_worktree_inherits_dirty_main_state_and_integrates_delta(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    (repository / 'shared.txt').write_text('user draft\n', encoding='utf-8')
    manager = SubagentWorktreeManager(repository)

    lease = manager.create('worker')

    assert (lease.path / 'shared.txt').read_text(
        encoding='utf-8'
    ) == 'user draft\n'
    (lease.path / 'shared.txt').write_text('agent result\n', encoding='utf-8')
    result = manager.integrate(lease)

    assert result.success is True
    assert result.integrated_paths == ('shared.txt',)
    assert (repository / 'shared.txt').read_text(
        encoding='utf-8'
    ) == 'agent result\n'


def test_concurrent_integrations_serialize_and_preserve_one_winner(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manager = SubagentWorktreeManager(repository)
    first = manager.create('agent-a')
    second = manager.create('agent-b')
    (first.path / 'shared.txt').write_text('agent a\n', encoding='utf-8')
    (second.path / 'shared.txt').write_text('agent b\n', encoding='utf-8')

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(manager.integrate, (first, second)))

    successes = [result for result in results if result.success]
    conflicts = [result for result in results if not result.success]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].conflicts == ('shared.txt',)
    assert (repository / 'shared.txt').read_text(encoding='utf-8') in {
        'agent a\n',
        'agent b\n',
    }


def test_unchanged_worktree_is_cleaned_up(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manager = SubagentWorktreeManager(repository)
    lease = manager.create('reader')

    result = manager.integrate(lease)

    assert result.changed_paths == ()
    assert result.cleaned_up is True
    assert not lease.path.exists()


def test_worktree_integrates_created_and_deleted_files(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manager = SubagentWorktreeManager(repository)
    lease = manager.create('worker')
    (lease.path / 'shared.txt').unlink()
    (lease.path / 'created.txt').write_text('new\n', encoding='utf-8')

    result = manager.integrate(lease)

    assert result.success is True
    assert result.integrated_paths == ('created.txt', 'shared.txt')
    assert not (repository / 'shared.txt').exists()
    assert (repository / 'created.txt').read_text(
        encoding='utf-8'
    ) == 'new\n'


def test_worktree_requires_git_repository(tmp_path: Path) -> None:
    manager = SubagentWorktreeManager(tmp_path)

    with pytest.raises(WorktreeError, match='Git repository'):
        manager.create('worker')


def create_repository(root: Path) -> Path:
    run(root, 'git', 'init')
    run(root, 'git', 'config', 'user.email', 'forge@example.com')
    run(root, 'git', 'config', 'user.name', 'Forge Test')
    (root / '.gitignore').write_text(
        '**/.forge/worktrees/\n',
        encoding='utf-8',
    )
    (root / 'shared.txt').write_text('base\n', encoding='utf-8')
    run(root, 'git', 'add', '.gitignore', 'shared.txt')
    run(root, 'git', 'commit', '-m', 'initial')
    return root


def run(root: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
