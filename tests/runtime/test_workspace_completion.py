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
from forge.runtime.agent_tool_calls import early_mutation_relevance_failure
from forge.runtime.state import VerificationEvidence
from forge.runtime.state import ToolCall
from forge.runtime.task_scope import evaluate_change_relevance, infer_task_scope
from forge.runtime.verification import verification_artifact_scope
from forge.runtime.workspace import WorkspaceTracker, should_skip_workspace_path
from forge.tasks.manager import TaskManager
from forge.tools.filesystem import CreateDirectoryTool


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


def test_prose_scope_hints_do_not_create_path_constraints() -> None:
    scope = infer_task_scope(
        '开始落地实现',
        scope_hints=(
            '当前目录为空项目，仅有 task.md',
            '需要从零创建完整项目',
        ),
    )

    assert scope.constrained is False
    assert scope.patterns == ()


def test_explicit_scope_globs_are_preserved() -> None:
    scope = infer_task_scope(
        '使用 Phaser 创建游戏',
        scope_hints=('src/game/**', 'tests/**'),
    )

    assert 'src/game/**' in scope.patterns
    assert 'tests/**' in scope.patterns
    assert evaluate_change_relevance(
        ('src/game/scenes/MainScene.ts',),
        scope,
    ).relevant
    assert evaluate_change_relevance(
        ('src/game/entities/Player.ts',),
        scope,
    ).relevant
    assert evaluate_change_relevance(('tests/game.test.ts',), scope).relevant


def test_paths_can_be_extracted_from_mixed_scope_hint_text() -> None:
    scope = infer_task_scope(
        '实现游戏',
        scope_hints=('主要修改 src/game/** 和 tests/**',),
    )

    assert 'src/game/**' in scope.patterns
    assert 'tests/**' in scope.patterns


def test_invalid_scope_hints_leave_scope_unconstrained() -> None:
    scope = infer_task_scope(
        '开始落地实现',
        scope_hints=('需要从零创建完整项目', '../outside', '/tmp/outside'),
    )

    assert scope.constrained is False


def test_game_scaffold_paths_are_allowed_for_game_goal() -> None:
    scope = infer_task_scope('使用 Phaser 创建游戏')
    paths = (
        'package.json',
        'tsconfig.json',
        'vite.config.ts',
        'index.html',
        'src/game/scenes/MainScene.ts',
        'src/game/entities/Player.ts',
        'src/game/systems/CollisionSystem.ts',
        'src/game/configs/game.ts',
        'tests/game.test.ts',
        'public/assets/player.png',
        'assets/sfx.wav',
    )

    assert all(
        evaluate_change_relevance((path,), scope).relevant
        for path in paths
    )


def test_explicit_allowed_paths_still_restrict_unrelated_targets() -> None:
    scope = infer_task_scope(
        '修复 sample.txt',
        scope_hints=('sample.txt',),
        scope_hint_source='allowed_path',
    )

    assert evaluate_change_relevance(('sample.txt',), scope).relevant
    assert not evaluate_change_relevance(('user.txt',), scope).relevant


def test_create_directory_allowed_when_only_scope_hints_are_prose() -> None:
    scope = infer_task_scope(
        '开始落地实现',
        scope_hints=(
            '当前目录为空项目，仅有 task.md',
            '需要从零创建完整项目',
        ),
    )

    result = early_mutation_relevance_failure(
        ToolCall(
            0,
            'mkdir-scenes',
            'create_directory',
            {'path': 'src/game/scenes'},
        ),
        tool_effect='workspace_write',
        change_required=True,
        task_scope_patterns=scope.patterns,
        task_scope_sources=scope.source_labels,
    )

    assert result is None


def test_create_directory_rejected_by_real_explicit_scope() -> None:
    scope = infer_task_scope(
        '修复 sample.txt',
        scope_hints=('sample.txt',),
        scope_hint_source='allowed_path',
    )

    result = early_mutation_relevance_failure(
        ToolCall(
            0,
            'mkdir-scenes',
            'create_directory',
            {'path': 'src/game/scenes'},
        ),
        tool_effect='workspace_write',
        change_required=True,
        task_scope_patterns=scope.patterns,
        task_scope_sources=scope.source_labels,
    )

    assert result is not None
    assert result.error is not None
    assert result.error.code == 'irrelevant_mutation_target'


def test_create_directory_still_rejects_workspace_escape(
    tmp_path: Path,
) -> None:
    result = run(
        CreateDirectoryTool(tmp_path).run(
            {'path': '../outside-forge-test', 'parents': True}
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'path_outside_repository'


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


def _verified_build_evidence(
    tracker: WorkspaceTracker,
    *,
    command: str = 'npx vite build',
    status: str = 'passed',
    exit_code: int = 0,
    timed_out: bool = False,
    generated_paths: tuple[str, ...] = (),
    cache_paths: tuple[str, ...] = (),
    side_effect_paths: tuple[str, ...] = (),
) -> VerificationEvidence:
    generated = generated_paths or tuple(
        path
        for path in tracker.filesystem_changed_paths
        if path.startswith(('dist/', 'release/'))
    )
    return VerificationEvidence(
        command=command,
        cwd='.',
        exit_code=exit_code,
        duration_seconds=0.1,
        timed_out=timed_out,
        workspace_revision=tracker.source_revision,
        source_revision=tracker.source_revision,
        filesystem_revision=tracker.filesystem_revision,
        status=status,
        verification_type='build',
        generated_artifact_paths=generated,
        cache_paths=cache_paths,
        verification_side_effect_paths=side_effect_paths,
        generated_artifact_fingerprints=tuple(
            (path, tracker.current.files[path])
            for path in generated
            if path in tracker.current.files
        ),
        cache_fingerprints=tuple(
            (path, tracker.current.files[path])
            for path in cache_paths
            if path in tracker.current.files
        ),
    )


def _checker_for_game_task(
    root: Path,
    tracker: WorkspaceTracker,
) -> CompletionChecker:
    task_manager = TaskManager(root)
    task_manager.begin_turn('实现 Phaser 游戏主入口 src/main.ts')
    return CompletionChecker(tracker, CompletionGate(root), task_manager)


def _prepare_vite_change(tmp_path: Path) -> tuple[WorkspaceTracker, Path]:
    initialize_git_repository(tmp_path)
    (tmp_path / 'package.json').write_text(
        '{"scripts":{"build":"vite build"}}\n',
        encoding='utf-8',
    )
    source = tmp_path / 'src' / 'main.ts'
    source.parent.mkdir()
    source.write_text('export const value = 1;\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'vite baseline'],
        cwd=tmp_path,
        check=True,
    )
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    source.write_text('export const value = 2;\n', encoding='utf-8')
    run(tracker.refresh())
    return tracker, source


def _write_build_outputs(root: Path, directory: str = 'dist') -> tuple[str, ...]:
    html = root / directory / 'index.html'
    bundle = root / directory / 'assets' / 'app.js'
    bundle.parent.mkdir(parents=True)
    html.write_text('<div id="app"></div>\n', encoding='utf-8')
    bundle.write_text('console.log("bundle");\n', encoding='utf-8')
    return (
        f'{directory}/assets/app.js',
        f'{directory}/index.html',
    )


def test_current_successful_vite_build_artifacts_do_not_block_completion(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert tracker.changed_paths == ('src/main.ts',)
    assert decision.allowed is True


def test_generated_artifacts_alone_do_not_satisfy_change_task(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    run(tracker.begin_turn())
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
        )
    )

    assert tracker.changed_paths == ()
    assert decision.allowed is False
    assert any('final Diff is empty' in reason for reason in decision.reasons)


def test_failed_build_artifacts_still_block_completion(tmp_path: Path) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(
        tracker,
        status='failed',
        exit_code=1,
        generated_paths=outputs,
    )

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('latest verification failed' in item for item in decision.reasons)
    assert any('outside the task deliverables' in item for item in decision.reasons)


def test_timed_out_build_artifacts_still_block_completion(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(
        tracker,
        status='timed_out',
        exit_code=-1,
        timed_out=True,
        generated_paths=outputs,
    )

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('timed out' in item for item in decision.reasons)
    assert any('outside the task deliverables' in item for item in decision.reasons)


def test_stale_build_artifacts_are_not_trusted(tmp_path: Path) -> None:
    tracker, source = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)
    source.write_text('export const value = 3;\n', encoding='utf-8')
    run(tracker.refresh())

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('changed after verification' in item for item in decision.reasons)
    assert any('outside the task deliverables' in item for item in decision.reasons)


def test_undeclared_generated_output_still_blocks_completion(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    declared = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=declared)
    (tmp_path / 'dist' / 'manual.js').write_text('manual\n', encoding='utf-8')
    run(tracker.refresh())

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('dist/manual.js' in item for item in decision.reasons)


def test_verification_source_side_effect_still_blocks_completion(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    side_effect = tmp_path / 'tmp' / 'debug.log'
    side_effect.parent.mkdir()
    side_effect.write_text('debug\n', encoding='utf-8')
    change = run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    assert change is not None
    evidence = _verified_build_evidence(
        tracker,
        status='failed',
        exit_code=1,
        generated_paths=outputs,
        side_effect_paths=change.classification.verification_side_effect_paths,
    )

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('Verification modified undeclared' in item for item in decision.reasons)


def test_forbidden_path_is_never_trusted_as_verification_output(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    hidden = tmp_path / 'tests' / 'hidden' / 'bundle.js'
    hidden.parent.mkdir(parents=True)
    hidden.write_text('forbidden\n', encoding='utf-8')
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(
        tracker,
        generated_paths=('tests/hidden/bundle.js',),
    )

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('Forbidden paths' in item for item in decision.reasons)


def test_verified_artifact_modified_after_build_is_not_trusted(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)
    (tmp_path / 'dist' / 'index.html').write_text('tampered\n', encoding='utf-8')
    run(tracker.refresh())

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is False
    assert any('dist/index.html' in item for item in decision.reasons)


def test_verified_artifact_unchanged_after_build_is_trusted(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert decision.allowed is True


def test_deleted_verified_artifact_does_not_block_completion(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    outputs = _write_build_outputs(tmp_path)
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(tracker, generated_paths=outputs)
    (tmp_path / 'dist' / 'index.html').unlink()
    (tmp_path / 'dist' / 'assets' / 'app.js').unlink()
    run(tracker.refresh())

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts',),
        )
    )

    assert tracker.filesystem_changed_paths == ('src/main.ts',)
    assert decision.allowed is True


def test_custom_vite_out_dir_is_allowed_as_verified_output(
    tmp_path: Path,
) -> None:
    tracker, _ = _prepare_vite_change(tmp_path)
    (tmp_path / 'vite.config.ts').write_text(
        "export default { build: { outDir: 'release' } };\n",
        encoding='utf-8',
    )
    run(tracker.refresh())
    outputs = _write_build_outputs(tmp_path, 'release')
    run(
        tracker.refresh(
            origin='verification',
            artifact_scope=verification_artifact_scope(
                'npx vite build',
                root=tmp_path,
                target='build',
            ),
        )
    )
    evidence = _verified_build_evidence(
        tracker,
        generated_paths=outputs,
    )

    decision = run(
        _checker_for_game_task(tmp_path, tracker).evaluate(
            evidence,
            mutation_attempted=True,
            evidence_paths=('src/main.ts', 'vite.config.ts'),
        )
    )

    assert decision.allowed is True


def test_new_npm_project_accepts_package_lock_as_supporting_config() -> None:
    scope = infer_task_scope(
        '创建 npm 项目',
        scope_hints=('package.json',),
        scope_hint_source='allowed_path',
    )

    relevance = evaluate_change_relevance(
        ('package.json', 'package-lock.json'),
        scope,
    )

    assert relevance.relevant is True


def test_lockfile_does_not_expand_scope_to_unrelated_paths() -> None:
    scope = infer_task_scope(
        '创建 npm 项目',
        scope_hints=('package.json',),
        scope_hint_source='allowed_path',
    )

    relevance = evaluate_change_relevance(
        ('package.json', 'package-lock.json', 'notes/debug.txt'),
        scope,
    )

    assert relevance.relevant is False
    assert 'notes/debug.txt' in relevance.reasons[0]
