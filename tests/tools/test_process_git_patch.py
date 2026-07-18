'''Tests for shell, Git, and patch tools against temporary repositories.'''

import asyncio
from pathlib import Path
import subprocess
import sys

from forge.tools.base import ToolResult
from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.patch import ApplyPatchTool
from forge.tools.shell import RunCommandTool
from forge.tools.verify import VerifyTool
from forge.runtime.workspace import WorkspaceTracker


def run(coroutine: object) -> ToolResult:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def initialize_git_repository(root: Path) -> None:
    subprocess.run(
        ['git', 'init', '--quiet'],
        cwd=root,
        check=True,
        capture_output=True,
    )
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
    subprocess.run(['git', 'add', 'sample.txt'], cwd=root, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'baseline'],
        cwd=root,
        check=True,
    )


def test_run_command_returns_stdout_stderr_exit_code_and_duration(
    tmp_path: Path,
) -> None:
    command = subprocess.list2cmdline(
        [
            sys.executable,
            '-c',
            'import sys; print("out"); print("err", file=sys.stderr)',
        ]
    )

    result = run(RunCommandTool(tmp_path).run({'command': command}))

    assert result.success is True
    assert result.metadata['exit_code'] == 0
    assert result.metadata['stdout'].strip() == 'out'
    assert result.metadata['stderr'].strip() == 'err'
    assert result.metadata['duration_seconds'] >= 0
    assert 'stdout:\nout' in result.content
    assert 'stderr:\nerr' in result.content


def test_run_command_returns_nonzero_exit_as_structured_error(
    tmp_path: Path,
) -> None:
    command = subprocess.list2cmdline(
        [sys.executable, '-c', 'raise SystemExit(7)']
    )

    result = run(RunCommandTool(tmp_path).run({'command': command}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'command_failed'
    assert result.metadata['exit_code'] == 7


def test_run_command_allows_stderr_merge_redirection(tmp_path: Path) -> None:
    redirect = '2' + chr(62) + chr(38) + '1'
    result = run(
        RunCommandTool(tmp_path).run(
            {'command': f'echo inspected {redirect}'}
        )
    )

    assert result.success is True


def test_verify_returns_revision_bound_evidence(tmp_path: Path) -> None:
    denied_commands = [
        'node -e writeFileSync(',
        'python -c Path.write_text(',
        'powershell Set-Content marker',
        'echo changed ' + chr(62) + ' game.html',
    ]
    for denied_command in denied_commands:
        denied = run(
            RunCommandTool(tmp_path).run({'command': denied_command})
        )
        assert denied.success is False
        assert denied.error is not None
        assert denied.error.code == 'shell_file_write_denied'

    initialize_git_repository(tmp_path)
    tracker = WorkspaceTracker(tmp_path)
    asyncio.run(tracker.begin_turn())
    script = 'print(' + repr('verified') + ')'
    command = subprocess.list2cmdline([sys.executable, '-c', script])

    result = run(VerifyTool(tmp_path, tracker).run({'command': command}))

    assert result.success is True
    assert result.metadata['verification'] is True
    assert result.metadata['workspace_revision'] == 0
    assert result.metadata['exit_code'] == 0
    assert 'verified' in result.content


def test_verify_failure_is_structured(tmp_path: Path) -> None:
    tracker = WorkspaceTracker(tmp_path)
    asyncio.run(tracker.begin_turn())
    command = subprocess.list2cmdline(
        [sys.executable, '-c', 'raise SystemExit(3)']
    )

    result = run(VerifyTool(tmp_path, tracker).run({'command': command}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'verification_failed'
    assert result.metadata['exit_code'] == 3


def test_git_status_and_diff_return_real_working_tree_state(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'sample.txt').write_text('changed\n', encoding='utf-8')

    status = run(GitStatusTool(tmp_path).run({}))
    diff = run(GitDiffTool(tmp_path).run({'path': 'sample.txt'}))

    assert status.success is True
    assert ' M sample.txt' in status.content
    assert diff.success is True
    assert '-old' in diff.content
    assert '+changed' in diff.content


def test_apply_patch_changes_the_file_and_reports_status(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    patch = (
        'diff --git a/sample.txt b/sample.txt\n'
        '--- a/sample.txt\n'
        '+++ b/sample.txt\n'
        '@@ -1 +1 @@\n'
        '-old\n'
        '+new\n'
    )

    result = run(ApplyPatchTool(tmp_path).run({'patch': patch}))

    assert result.success is True
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'new\n'
    assert result.metadata['changed_files'] == ['sample.txt']
    assert 'M sample.txt' in result.content


def test_apply_patch_rejection_does_not_modify_files(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)

    result = run(
        ApplyPatchTool(tmp_path).run(
            {
                'patch': (
                    '--- a/sample.txt\n'
                    '+++ b/sample.txt\n'
                    '@@ -1 +1 @@\n'
                    '-not-the-current-content\n'
                    '+new\n'
                )
            }
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'patch_rejected'
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'old\n'


def test_apply_patch_description_requires_small_focused_writes(
    tmp_path: Path,
) -> None:
    description = ApplyPatchTool(tmp_path).definition['description']

    assert 'limited to 8000 characters' in description
    assert 'split large HTML' in description
    assert 'Use write_file only for small full-file content' in description


def test_apply_patch_rejects_payload_over_8000_characters(
    tmp_path: Path,
) -> None:
    result = run(ApplyPatchTool(tmp_path).run({'patch': 'x' * 8_001}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
