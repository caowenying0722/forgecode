'''Local command execution tool and shared subprocess helpers.'''

from __future__ import annotations

import os
from pathlib import Path
import re
from time import time
from typing import TYPE_CHECKING

from pydantic import Field

from forge.runtime.verification import (
    NON_INTERACTIVE_ENV,
    VerificationTransaction,
    classify_verification_command,
    discover_validation_commands,
    verification_artifact_scope,
    verification_cache_key,
)
from forge.runtime.process import (
    ProcessResult,
    process_metadata,
    render_process_output,
    run_process,
)

from forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)

if TYPE_CHECKING:
    from forge.runtime.background import BackgroundTaskManager
    from forge.runtime.verification_ledger import VerificationLedger
    from forge.runtime.workspace import WorkspaceTracker


class RunCommandInput(ToolInput):
    command: str = Field(min_length=1)
    cwd: str = '.'
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    stdin: str | None = Field(default=None, max_length=8_000)
    run_in_background: bool = False


class RunCommandTool(Tool[RunCommandInput]):
    name = 'run_command'
    description = (
        'Run an executable repository command for exploration, diagnostics, '
        'or development. Do not use it to display source files or directory '
        'trees; use read_file, grep, find_files, or list_directory. Do not '
        'write files through scripts or redirection; use write_file, '
        'replace_text, or apply_patch. Use verify instead when the command is '
        'intended as formal completion evidence. Set run_in_background=true '
        'only for slow commands where useful work can continue while the '
        'command runs; completion will be injected later as a '
        'task_notification. For multiline scripts, pass command="python -" '
        'or command="node" and put the script in stdin; do not embed a POSIX '
        'heredoc in command. '
        + (
            'Commands run through Windows cmd.exe, which does not support '
            'the POSIX << heredoc syntax.'
            if os.name == 'nt'
            else 'Commands run through the platform default shell.'
        )
    )
    input_model = RunCommandInput
    effect = 'process'

    def __init__(
        self,
        root: Path,
        background_manager: 'BackgroundTaskManager | None' = None,
        workspace_tracker: 'WorkspaceTracker | None' = None,
        verification_ledger: 'VerificationLedger | None' = None,
    ) -> None:
        super().__init__(root)
        self.background_manager = background_manager
        self.workspace_tracker = workspace_tracker
        self.verification_ledger = verification_ledger

    async def execute(self, arguments: RunCommandInput) -> ToolResult:
        if os.name == 'nt' and has_unquoted_heredoc(arguments.command):
            raise ToolExecutionError(
                'unsupported_shell_syntax',
                'Windows cmd.exe does not support POSIX << heredocs. Use '
                'command="python -" or command="node" and pass the '
                'multiline program in the stdin field.',
                details={
                    'shell': 'cmd.exe',
                    'supported_fields': [
                        'command',
                        'cwd',
                        'timeout_seconds',
                        'stdin',
                    ],
                },
            )
        read_reason = shell_file_read_reason(arguments.command)
        if read_reason is not None:
            raise ToolExecutionError(
                'shell_file_read_denied',
                'run_command cannot be used as a substitute for repository '
                'reading tools. Use read_file, list_directory, grep, or '
                'find_files so ForgeCode can track the evidence.',
                details={'detected': read_reason},
            )
        denied_reason = shell_file_write_reason(arguments.command)
        if denied_reason is not None:
            raise ToolExecutionError(
                'shell_file_write_denied',
                'run_command cannot be used to write repository files. '
                'Use write_file, replace_text, or apply_patch instead.',
                details={'detected': denied_reason},
            )
        if arguments.stdin is not None:
            stdin_read_reason = shell_file_read_reason(arguments.stdin)
            if stdin_read_reason is not None:
                raise ToolExecutionError(
                    'shell_file_read_denied',
                    'run_command stdin cannot bypass repository reading '
                    'tools. Use read_file, list_directory, grep, or find_files.',
                    details={'detected': stdin_read_reason},
                )
            stdin_write_reason = shell_file_write_reason(arguments.stdin)
            if stdin_write_reason is not None:
                raise ToolExecutionError(
                    'shell_file_write_denied',
                    'run_command stdin cannot write repository files. Use '
                    'write_file, replace_text, or apply_patch instead.',
                    details={'detected': stdin_write_reason},
                )
        cwd = resolve_repository_path(self.root, arguments.cwd)
        if not cwd.is_dir():
            raise ToolExecutionError(
                'not_a_directory',
                f'Command cwd is not a directory: {arguments.cwd}',
            )
        if arguments.run_in_background:
            if self.background_manager is None:
                raise ToolExecutionError(
                    'background_not_available',
                    'run_in_background is only available inside the main '
                    'ForgeCode conversation loop.',
                )
            background = self.background_manager.start_command(
                command=arguments.command,
                cwd=cwd,
                display_cwd=display_path(self.root, cwd),
                timeout_seconds=arguments.timeout_seconds,
                input_text=arguments.stdin,
            )
            return ToolResult.ok(
                f'Background command {background.id} started.',
                content=(
                    f'[Background task {background.id} started]\n'
                    f'Command: {arguments.command}\n'
                    'Result will be injected as a task_notification when '
                    'the command completes.'
                ),
                metadata={
                    'background_started': True,
                    'background_id': background.id,
                    'command': arguments.command,
                    'cwd': background.cwd,
                },
            )
        routed = await self._run_as_verification_if_applicable(arguments, cwd)
        if routed is not None:
            return routed
        result = await run_process(
            arguments.command,
            cwd=cwd,
            timeout_seconds=arguments.timeout_seconds,
            input_text=arguments.stdin,
            shell=True,
        )
        metadata = {
            **process_metadata(result),
            'command': arguments.command,
            'cwd': display_path(self.root, cwd),
            'stdin_characters': len(arguments.stdin or ''),
        }
        content = render_process_output(result)
        if result.timed_out:
            return ToolResult.fail(
                'command_timeout',
                f'Command timed out after {arguments.timeout_seconds:g}s.',
                content=content,
                metadata=metadata,
            )
        if result.exit_code != 0:
            return ToolResult.fail(
                'command_failed',
                f'Command exited with code {result.exit_code}.',
                content=content,
                metadata=metadata,
            )
        return ToolResult.ok(
            f'Command completed with exit code 0 in '
            f'{result.duration_seconds:.3f}s.',
            content=content,
            metadata=metadata,
        )

    async def _run_as_verification_if_applicable(
        self,
        arguments: RunCommandInput,
        cwd: Path,
    ) -> ToolResult | None:
        if (
            self.workspace_tracker is None
            or self.verification_ledger is None
            or arguments.stdin is not None
        ):
            return None
        discovered = discover_validation_commands(self.root)
        status, _ = classify_verification_command(
            arguments.command,
            discovered_commands=discovered,
        )
        if status != 'passed':
            return None
        command = arguments.command.strip()
        display_cwd = display_path(self.root, cwd)
        source_revision = self.workspace_tracker.source_revision
        filesystem_revision = self.workspace_tracker.filesystem_revision
        scope = verification_artifact_scope(
            command,
            root=self.root,
            target='auto',
        )
        key = verification_cache_key(
            source_revision=source_revision,
            command=command,
            cwd=display_cwd,
            scope=scope,
        )
        reusable = self.verification_ledger.reusable(key)
        if reusable is not None:
            result = ToolResult.ok(
                f'Reused verification evidence for source revision '
                f'{source_revision}.',
                content=reusable.stdout_stderr_summary,
                metadata={
                    'verification': True,
                    'verification_reused': True,
                    'verification_status': reusable.status,
                    'verification_type': reusable.kind,
                    'command': reusable.command,
                    'command_id': reusable.target,
                    'cwd': reusable.cwd,
                    'workspace_revision': source_revision,
                    'source_revision': source_revision,
                    'filesystem_revision': filesystem_revision,
                    'exit_code': reusable.exit_code,
                    'duration_seconds': reusable.duration_seconds,
                    'timed_out': reusable.timed_out,
                    'verification_side_effect_paths': list(
                        reusable.side_effect_paths
                    ),
                    'generated_artifact_paths': list(
                        reusable.generated_artifact_paths
                    ),
                    'cache_paths': list(reusable.cache_paths),
                    'generated_artifact_fingerprints': [
                        list(item)
                        for item in reusable.generated_artifact_fingerprints
                    ],
                    'cache_fingerprints': [
                        list(item) for item in reusable.cache_fingerprints
                    ],
                    'verification_ledger_recorded': True,
                },
            )
            self.verification_ledger.record_from_metadata(
                result.metadata,
                content=result.content,
                evidence_source='cache',
                reusable_key=key,
            )
            return result
        started_at = time()
        process = await run_process(
            command,
            cwd=cwd,
            timeout_seconds=arguments.timeout_seconds,
            shell=True,
            env=NON_INTERACTIVE_ENV,
        )
        change = await self.workspace_tracker.refresh(
            origin='verification',
            artifact_scope=scope,
        )
        if change is None:
            raise ToolExecutionError(
                'workspace_snapshot_unavailable',
                'Verification completed, but the workspace delta could not '
                'be captured.',
            )
        transaction = VerificationTransaction.from_workspace_change(
            command=command,
            cwd=display_cwd,
            source_revision_before=source_revision,
            filesystem_revision_before=filesystem_revision,
            change=change,
        )
        classification = transaction.classification
        verification_status = (
            'timed_out'
            if process.timed_out
            else ('passed' if process.exit_code == 0 else 'failed')
        )
        side_effect_paths = classification.verification_side_effect_paths
        if side_effect_paths and verification_status == 'passed':
            verification_status = 'failed'
        generated_artifact_fingerprints = _current_fingerprints(
            self.workspace_tracker,
            classification.generated_paths,
        )
        cache_fingerprints = _current_fingerprints(
            self.workspace_tracker,
            classification.cache_paths,
        )
        metadata = {
            **process_metadata(process),
            'command': command,
            'cwd': display_cwd,
            'stdin_characters': 0,
            'verification': True,
            'verification_status': verification_status,
            'verification_type': scope.verification_type,
            'command_id': 'run_command',
            'workspace_revision': source_revision,
            'source_revision': source_revision,
            'filesystem_revision': self.workspace_tracker.filesystem_revision,
            'verification_artifact_scope': [
                {
                    'pattern': rule.pattern,
                    'kind': rule.kind,
                    'description': rule.description,
                }
                for rule in scope.allowed_writes
            ],
            'verification_side_effect_paths': list(side_effect_paths),
            'generated_artifact_paths': list(classification.generated_paths),
            'cache_paths': list(classification.cache_paths),
            'generated_artifact_fingerprints': [
                list(item) for item in generated_artifact_fingerprints
            ],
            'cache_fingerprints': [list(item) for item in cache_fingerprints],
            'verification_transaction': transaction.as_metadata(),
            'verification_ledger_recorded': True,
        }
        content = render_process_output(process)
        self.verification_ledger.record_from_metadata(
            metadata,
            content=content,
            evidence_source='run_command',
            reusable_key=key,
            started_at=started_at,
            finished_at=time(),
        )
        if process.timed_out:
            return ToolResult.fail(
                'verification_timeout',
                f'Verification timed out after {arguments.timeout_seconds:g}s.',
                content=content,
                metadata=metadata,
            )
        if process.exit_code != 0:
            return ToolResult.fail(
                'verification_failed',
                f'Verification exited with code {process.exit_code}.',
                content=content,
                metadata=metadata,
            )
        if side_effect_paths:
            return ToolResult.fail(
                'verification_side_effect',
                'Verification modified undeclared source or workspace paths: '
                + ', '.join(side_effect_paths),
                content=content,
                metadata=metadata,
            )
        return ToolResult.ok(
            f'Verification passed in {process.duration_seconds:.3f}s.',
            content=content,
            metadata=metadata,
        )


def _current_fingerprints(
    tracker: 'WorkspaceTracker',
    paths: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, tracker.current.files[path])
        for path in paths
        if path in tracker.current.files
    )


SCRIPT_WRITE_PATTERNS = (
    (
        re.compile(
            r'\b(?:writeFile|writeFileSync|appendFile|appendFileSync)\s*\(',
            re.IGNORECASE,
        ),
        'Node filesystem write API',
    ),
    (
        re.compile(r'\.(?:write_text|write_bytes)\s*\(', re.IGNORECASE),
        'Python pathlib write API',
    ),
    (
        re.compile(
            r'\bopen\s*\([^\n]*,\s*[\x27\x22](?:w|a|x|\+)',
            re.IGNORECASE,
        ),
        'Python writable open mode',
    ),
    (
        re.compile(
            r'\b(?:Set-Content|Add-Content|Out-File)\b',
            re.IGNORECASE,
        ),
        'PowerShell file-writing command',
    ),
)


SCRIPT_READ_PATTERNS = (
    (re.compile(r'\bGet-Content\b', re.IGNORECASE), 'PowerShell Get-Content'),
    (re.compile(r'\bGet-ChildItem\b', re.IGNORECASE), 'PowerShell Get-ChildItem'),
    (re.compile(r'(^|[|;&]\s*)\b(?:cat|head|tail|nl)\b', re.IGNORECASE), 'shell file reader'),
    (re.compile(r'(^|[|;&]\s*)\bsed\s+-n\b', re.IGNORECASE), 'sed line reader'),
)


def shell_file_read_reason(command: str) -> str | None:
    '''Detect shell commands that bypass repository evidence tracking.'''
    for pattern, reason in SCRIPT_READ_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def shell_file_write_reason(command: str) -> str | None:
    '''Detect common direct file-writing shortcuts before shell execution.'''
    for pattern, reason in SCRIPT_WRITE_PATTERNS:
        if pattern.search(command):
            return reason
    if has_unquoted_output_redirection(command):
        return 'shell output redirection'
    return None


def has_unquoted_output_redirection(command: str) -> bool:
    single_quoted = False
    double_quoted = False
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if character == chr(39) and not double_quoted:
            single_quoted = not single_quoted
            continue
        if character == chr(34) and not single_quoted:
            double_quoted = not double_quoted
            continue
        if character != '>' or single_quoted or double_quoted:
            continue
        following = command[index + 1:index + 2]
        preceding = command[index - 1:index] if index else ''
        if following == '&' or preceding == '=':
            continue
        return True
    return False


def has_unquoted_heredoc(command: str) -> bool:
    '''Detect POSIX heredoc operators without matching quoted bit shifts.'''
    single_quoted = False
    double_quoted = False
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if character == chr(39) and not double_quoted:
            single_quoted = not single_quoted
            continue
        if character == chr(34) and not single_quoted:
            double_quoted = not double_quoted
            continue
        if (
            character == '<'
            and not single_quoted
            and not double_quoted
            and command[index + 1:index + 2] == '<'
        ):
            return True
    return False
