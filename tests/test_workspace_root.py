'''Tests for CLI workspace discovery independent of the ForgeCode source tree.'''

from pathlib import Path

import pytest

from forge.workspace_root import find_git_root, resolve_workspace


def test_workspace_discovers_parent_git_root(tmp_path: Path) -> None:
    (tmp_path / '.git').mkdir()
    nested = tmp_path / 'apps' / 'web'
    nested.mkdir(parents=True)

    location = resolve_workspace(cwd=nested, process_cwd=tmp_path)

    assert location.root == tmp_path.resolve()
    assert location.cwd == nested.resolve()
    assert location.source == 'git'


def test_workspace_can_keep_exact_cwd_without_git_discovery(tmp_path: Path) -> None:
    (tmp_path / '.git').mkdir()
    nested = tmp_path / 'src'
    nested.mkdir()

    location = resolve_workspace(
        cwd=nested,
        discover_git=False,
        process_cwd=tmp_path,
    )

    assert location.root == nested.resolve()
    assert location.source == 'cwd'


def test_explicit_root_must_contain_cwd(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    outside = tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()

    with pytest.raises(ValueError, match='outside'):
        resolve_workspace(root=root, cwd=outside, process_cwd=tmp_path)


def test_git_worktree_marker_file_is_recognized(tmp_path: Path) -> None:
    (tmp_path / '.git').write_text('gitdir: elsewhere\n', encoding='utf-8')
    nested = tmp_path / 'nested'
    nested.mkdir()

    assert find_git_root(nested) == tmp_path.resolve()
