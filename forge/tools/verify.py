'''Command verification that produces completion evidence.'''

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from forge.runtime.verification import (
    ValidationTarget,
    choose_validation_command,
    classify_verification_command,
    discover_validation_commands,
)
from forge.runtime.verification_executor import VerificationExecutor
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)
if TYPE_CHECKING:
    from forge.runtime.verification_ledger import VerificationLedger


class VerifyInput(ToolInput):
    target: ValidationTarget = 'auto'
    command_id: str = ''
    command: str = ''
    cwd: str = '.'
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)


class VerifyTool(Tool[VerifyInput]):
    name = 'verify'
    description = (
        'Run one non-interactive project validation command as formal '
        'completion evidence after workspace changes. Prefer target=auto, '
        'or choose target=build/test/lint/typecheck/diff or a discovered '
        'command_id. Do not use verify for environment probes, file discovery, '
        'interactive dev servers, or arbitrary shell commands. A successful '
        'result applies only to the exact current workspace revision.'
    )
    input_model = VerifyInput
    effect = 'process'

    def __init__(
        self,
        root: Path,
        tracker: WorkspaceTracker,
        ledger: 'VerificationLedger | None' = None,
    ) -> None:
        super().__init__(root)
        self.tracker = tracker
        self.ledger = ledger

    async def execute(self, arguments: VerifyInput) -> ToolResult:
        discovered = discover_validation_commands(self.root)
        filesystem_revision = self.tracker.filesystem_revision
        source_revision = self.tracker.source_revision
        selected = None
        if arguments.command.strip():
            status, reason = classify_verification_command(
                arguments.command,
                discovered_commands=discovered,
            )
            if status == 'invalid':
                return ToolResult.fail(
                    'verification_command_invalid',
                    reason,
                    metadata={
                        'verification': True,
                        'verification_status': 'invalid',
                        'command': arguments.command.strip(),
                        'command_id': 'invalid',
                        'cwd': arguments.cwd,
                        'workspace_revision': source_revision,
                        'source_revision': source_revision,
                        'filesystem_revision': filesystem_revision,
                        'exit_code': -1,
                        'duration_seconds': 0.0,
                        'timed_out': False,
                        'stderr': reason,
                        'available_validation_commands': [
                            {
                                'id': command.id,
                                'command': command.command,
                                'target': command.target,
                                'source': command.source,
                            }
                            for command in discovered
                        ],
                    },
                )
            command = arguments.command.strip()
            cwd_argument = arguments.cwd
            command_id = 'custom'
        else:
            selected = choose_validation_command(
                self.root,
                target=arguments.target,
                command_id=arguments.command_id.strip(),
            )
            if selected is None:
                return ToolResult.fail(
                    'verification_unavailable',
                    'No project validation command is available.',
                    metadata={
                        'verification': True,
                        'verification_status': 'unavailable',
                        'command': '',
                        'command_id': arguments.command_id.strip(),
                        'cwd': arguments.cwd,
                        'workspace_revision': source_revision,
                        'source_revision': source_revision,
                        'filesystem_revision': filesystem_revision,
                        'exit_code': -1,
                        'duration_seconds': 0.0,
                        'timed_out': False,
                        'stderr': 'No project validation command is available.',
                    },
                )
            command = selected.command
            cwd_argument = selected.cwd
            command_id = selected.id

        cwd = resolve_repository_path(self.root, cwd_argument)
        if not cwd.is_dir():
            raise ToolExecutionError(
                'not_a_directory',
                f'Verification cwd is not a directory: {cwd_argument}',
            )
        return await VerificationExecutor(
            self.root,
            self.tracker,
            self.ledger,
        ).execute(
            command=command,
            cwd=cwd,
            display_cwd=display_path(self.root, cwd),
            timeout_seconds=arguments.timeout_seconds,
            command_id=command_id,
            target=arguments.target,
            evidence_source='verify',
            discovered_commands=discovered,
        )
