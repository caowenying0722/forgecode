'''Tests for M2 workspace tracking and deterministic completion checks.'''

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from forge.runtime.completion import (
    CompletionGate,
    TaskPolicy,
    matches_any,
)
from forge.runtime.completion_checker import CompletionChecker
from forge.runtime.state import VerificationEvidence
from forge.runtime.workspace import WorkspaceTracker, should_skip_workspace_path
from forge.tasks.manager import TaskManager


def initialize_git_repository(root: Path) -> None:
    subprocess.run(['git', 'init', '--quiet'], cwd=root, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'forge@example.test'],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'ForgeCode Tests'],
        cwd=root,
        check=True,
    )
    (root / 'sample.txt').write_text('old\n', encoding='utf-8')
    (root / 'user.txt').write_text('baseline\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'baseline'],
        cwd=root,
        check=True,
    )


def test_workspace_tracker_imports_in_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'from forge.runtime.workspace import WorkspaceTracker; '
                'print(WorkspaceTracker.__name__)'
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'WorkspaceTracker'


def test_workspace_tracker_initializes_private_gitdir_for_non_git_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    task = tmp_path / 'task.md'
    task.write_text('existing task\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)

    run(tracker.begin_turn())

    git_marker = tmp_path / '.git'
    assert tracker.available is True
    assert git_marker.is_file()
    assert git_marker.read_text(encoding='utf-8').startswith('gitdir: ')
    assert 'task.md' in tracker.baseline.files
    assert not any(path.startswith('.forge/') for path in tracker.current.files)
    head = subprocess.run(
        ['git', 'rev-parse', '--verify', 'HEAD'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert head.returncode == 0

    generated = tmp_path / 'src' / 'main.ts'
    generated.parent.mkdir()
    generated.write_text('export {};\n', encoding='utf-8')
    change = run(tracker.refresh())

    assert change is not None
    assert change.paths == ('src/main.ts',)
    assert tracker.changed_paths == ('src/main.ts',)


def run(coroutine: object) -> Any:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_deprecated_require_changes_does_not_create_code_task(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    decision = run(
        CompletionGate(
            tmp_path,
            TaskPolicy(require_changes=True),
        ).evaluate(
            tracker,
            None,
            mutation_attempted=False,
        )
    )

    assert decision.allowed is True


def test_workspace_tracker_preserves_preexisting_user_changes(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'user.txt').write_text('user edit\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)

    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('agent edit\n', encoding='utf-8')
    change = run(tracker.refresh())

    assert change is not None
    assert change.revision == 1
    assert change.paths == ('sample.txt',)
    assert tracker.changed_paths == ('sample.txt',)


def test_path_patterns_match_deep_source_files() -> None:
    assert matches_any('src/todo.ts', ('src/**',))
    assert matches_any('src/main/java/Order.java', ('src/main/**',))
    assert matches_any('tests/hidden/a/b.py', ('tests/hidden/**',))


def test_workspace_tracker_detects_untracked_files_and_reverts(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())

    generated = tmp_path / 'generated.txt'
    generated.write_text('new\n', encoding='utf-8')
    first = run(tracker.refresh())
    generated.unlink()
    second = run(tracker.refresh())

    assert first is not None and first.revision == 1
    assert first.paths == ('generated.txt',)
    assert second is not None and second.revision == 2
    assert tracker.changed_paths == ()


def test_workspace_tracker_ignores_untracked_local_caches(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    cache_file = tmp_path / '.cache' / 'uv' / 'wheel.whl'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b'cached wheel')
    tracker = WorkspaceTracker(tmp_path)

    run(tracker.begin_turn())

    assert should_skip_workspace_path('.cache/uv/wheel.whl')
    assert should_skip_workspace_path('.forge/logs/tools.jsonl')
    assert '.cache/uv/wheel.whl' not in tracker.current.files


def test_workspace_tracker_watches_ignored_write_targets(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / '.gitignore').write_text('ignored/\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.gitignore'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'ignore generated files'],
        cwd=tmp_path,
        check=True,
    )
    ignored = tmp_path / 'ignored'
    ignored.mkdir()
    existing = ignored / 'app.js'
    existing.write_text('old\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)

    run(tracker.begin_turn())
    tracker.watch_paths(('ignored/app.js', 'ignored/new.js'))
    existing.write_text('changed\n', encoding='utf-8')
    (ignored / 'new.js').write_text('created\n', encoding='utf-8')
    change = run(tracker.refresh())
    unchanged = run(tracker.refresh())

    assert change is not None
    assert change.revision == 1
    assert change.paths == ('ignored/app.js', 'ignored/new.js')
    assert tracker.changed_paths == ('ignored/app.js', 'ignored/new.js')
    assert unchanged is None


def test_workspace_tracker_watched_paths_can_include_local_caches(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    cache_file = tmp_path / '.cache' / 'generated.txt'
    cache_file.parent.mkdir()
    cache_file.write_text('old\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)

    run(tracker.begin_turn())
    tracker.watch_paths(('.cache/generated.txt',))
    cache_file.write_text('new\n', encoding='utf-8')
    change = run(tracker.refresh())

    assert change is not None
    assert change.paths == ('.cache/generated.txt',)
    assert tracker.changed_paths == ('.cache/generated.txt',)


def test_completion_gate_requires_verification_only_when_policy_requests_it(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())
    gate = CompletionGate(
        tmp_path,
        TaskPolicy(require_verification=True),
    )

    missing = run(gate.evaluate(tracker, None, mutation_attempted=False))
    current = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )
    accepted = run(
        gate.evaluate(tracker, current, mutation_attempted=False)
    )
    (tmp_path / 'sample.txt').write_text('changed again\n', encoding='utf-8')
    run(tracker.refresh())
    stale = run(gate.evaluate(tracker, current, mutation_attempted=False))

    assert missing.allowed is False
    assert 'has not been verified' in missing.reasons[0]
    assert accepted.allowed is True
    assert stale.allowed is False
    assert any('changed after verification' in item for item in stale.reasons)


def test_completion_gate_requires_current_verification_for_changes_by_default(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())

    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            None,
            mutation_attempted=False,
        )
    )

    assert decision.allowed is False
    assert 'has not been verified' in decision.reasons[0]


def test_completion_gate_requires_final_diff_review_when_review_state_is_known(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )

    missing_review = run(
        CompletionGate(
            tmp_path,
            TaskPolicy(require_diff_review=True),
        ).evaluate(
            tracker,
            evidence,
            mutation_attempted=True,
            reviewed_paths=set(),
        )
    )
    reviewed = run(
        CompletionGate(
            tmp_path,
            TaskPolicy(require_diff_review=True),
        ).evaluate(
            tracker,
            evidence,
            mutation_attempted=True,
            reviewed_paths={'sample.txt'},
        )
    )

    assert missing_review.allowed is False
    assert 'final Diff has not been reviewed' in missing_review.reasons[0]
    assert reviewed.allowed is True


def test_source_change_rejects_format_only_verification_when_tests_exist(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'pyproject.toml').write_text(
        '[tool.pytest.ini_options]\n', encoding='utf-8'
    )
    source = tmp_path / 'app.py'
    source.write_text('value = 1\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    source.write_text('value = 2\n', encoding='utf-8')
    run(tracker.refresh())
    weak = VerificationEvidence(
        command='git diff --check',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )
    strong = VerificationEvidence(
        command='python -m pytest -q',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )

    weak_decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker, weak, mutation_attempted=True
        )
    )
    strong_decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker, strong, mutation_attempted=True
        )
    )

    assert weak_decision.allowed is False
    assert 'verification command does not run' in weak_decision.reasons[0]
    assert strong_decision.allowed is True


def test_completion_gate_blocks_current_optional_verification_failure(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')
    run(tracker.refresh())
    failed = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=1,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )

    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            failed,
            mutation_attempted=False,
        )
    )

    assert decision.allowed is False
    assert 'latest verification failed' in decision.reasons[0]


def test_completion_gate_ignores_unrelated_preexisting_whitespace_errors(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'user.txt').write_text(
        'preexisting user edit with trailing spaces  \n',
        encoding='utf-8',
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'sample.txt').write_text('agent edit\n', encoding='utf-8')
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )

    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            evidence,
            mutation_attempted=True,
        )
    )

    assert tracker.changed_paths == ('sample.txt',)
    assert decision.allowed is True


def test_completion_gate_checks_task_local_change_to_untracked_file(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    untracked = tmp_path / 'play' / 'world.js'
    untracked.parent.mkdir()
    untracked.write_text('const face = 1;\n', encoding='utf-8')
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    untracked.write_text('const face = 6;  \n', encoding='utf-8')
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='git diff --check',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )

    decision = run(
        CompletionGate(tmp_path).evaluate(
            tracker,
            evidence,
            mutation_attempted=True,
        )
    )

    assert tracker.changed_paths == ('play/world.js',)
    assert decision.allowed is False
    assert any(
        'untracked file: play/world.js' in reason
        for reason in decision.reasons
    )


def test_completion_gate_rejects_failed_verification_and_empty_diff(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    failed = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=1,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=0,
    )
    gate = CompletionGate(
        tmp_path,
        TaskPolicy(require_changes=True, require_verification=True),
    )

    decision = run(
        gate.evaluate(tracker, failed, mutation_attempted=True)
    )

    assert decision.allowed is False
    assert any('final Diff is empty' in item for item in decision.reasons)
    assert any('verification failed' in item for item in decision.reasons)


def test_completion_gate_rejects_forbidden_and_out_of_scope_paths(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    hidden = tmp_path / 'tests' / 'hidden'
    hidden.mkdir(parents=True)
    (hidden / 'test_secret.py').write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'hidden baseline'],
        cwd=tmp_path,
        check=True,
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (hidden / 'test_secret.py').write_text('disabled\n', encoding='utf-8')
    (tmp_path / 'user.txt').write_text('outside\n', encoding='utf-8')
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='pytest',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )
    gate = CompletionGate(
        tmp_path,
        TaskPolicy(allowed_paths=('sample.txt',)),
    )

    decision = run(
        gate.evaluate(tracker, evidence, mutation_attempted=False)
    )

    assert decision.allowed is False
    assert any('Forbidden paths' in item for item in decision.reasons)
    assert any('outside the allowed scope' in item for item in decision.reasons)


def test_completion_checker_rejects_tmp_only_change_for_game_task(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    (tmp_path / 'tmp_check.txt').write_text('placeholder\n', encoding='utf-8')
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='python -m pytest -q',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )
    task_manager = TaskManager(tmp_path)
    task_manager.begin_turn('补齐 Phaser 游戏骨架，创建场景、Player、Enemy 和碰撞系统')
    checker = CompletionChecker(
        tracker,
        CompletionGate(tmp_path),
        task_manager,
    )

    decision = run(
        checker.evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=(),
        )
    )

    assert decision.allowed is False
    assert any('temporary' in item for item in decision.reasons)


def test_completion_checker_accepts_evidence_related_change(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    scene = tmp_path / 'src' / 'game' / 'scenes' / 'MainScene.ts'
    scene.parent.mkdir(parents=True)
    scene.write_text('export class MainScene {}\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'game baseline'],
        cwd=tmp_path,
        check=True,
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    scene.write_text(
        'export class MainScene { create() {} }\n',
        encoding='utf-8',
    )
    run(tracker.refresh())
    evidence = VerificationEvidence(
        command='npm run build --if-present',
        cwd='.',
        exit_code=0,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
    )
    task_manager = TaskManager(tmp_path)
    task_manager.begin_turn('实现游戏主场景')
    checker = CompletionChecker(
        tracker,
        CompletionGate(tmp_path),
        task_manager,
    )

    decision = run(
        checker.evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/game/scenes/MainScene.ts',),
        )
    )

    assert decision.allowed is True
