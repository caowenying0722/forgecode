'''Command verification that produces completion evidence.'''

from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from pydantic import Field

from forge.runtime.verification import (
    NON_INTERACTIVE_ENV,
    ValidationTarget,
    choose_validation_command,
    classify_verification_command,
    discover_validation_commands,
    verification_artifact_scope,
    verification_cache_key,
)
from forge.runtime.workspace import WorkspaceTracker
from forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)
from forge.runtime.process import (
    process_metadata,
    render_process_output,
    run_process,
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
        scope = verification_artifact_scope(
            command,
            root=self.root,
            target=arguments.target,
        )
        key = verification_cache_key(
            source_revision=source_revision,
            command=command,
            cwd=display_path(self.root, cwd),
            scope=scope,
        )
        cached = self.tracker.verification_cache.get(key)
        if isinstance(cached, ToolResult):
            reused = ToolResult.ok(
                f'Reused verification evidence for source revision '
                f'{source_revision}.',
                content=cached.content,
                metadata={
                    **cached.metadata,
                    'cache_hit': True,
                    'verification_reused': True,
                    'workspace_revision': source_revision,
                    'source_revision': source_revision,
                    'filesystem_revision': self.tracker.filesystem_revision,
                },
            )
            if self.ledger is not None:
                reused.metadata['verification_ledger_recorded'] = True
                self.ledger.record_from_metadata(
                    reused.metadata,
                    content=reused.content,
                    evidence_source='cache',
                    reusable_key=key,
                )
            return reused
        started_at = time()
        result = await run_process(
            command,
            cwd=cwd,
            timeout_seconds=arguments.timeout_seconds,
            shell=True,
            env=NON_INTERACTIVE_ENV,
        )
        change = await self.tracker.refresh(
            origin='verification',
            artifact_scope=scope,
        )
        classification = (
            change.classification
            if change is not None
            else self.tracker.last_classification
        )
        verification_status = (
            'timed_out'
            if result.timed_out
            else ('passed' if result.exit_code == 0 else 'failed')
        )
        side_effect_paths = classification.verification_side_effect_paths
        if side_effect_paths and verification_status == 'passed':
            verification_status = 'failed'
        generated_artifact_fingerprints = _current_fingerprints(
            self.tracker,
            classification.generated_paths,
        )
        cache_fingerprints = _current_fingerprints(
            self.tracker,
            classification.cache_paths,
        )
        metadata = {
            **process_metadata(result),
            'command': command,
            'command_id': command_id,
            'cwd': display_path(self.root, cwd),
            'workspace_revision': source_revision,
            'source_revision': source_revision,
            'filesystem_revision': self.tracker.filesystem_revision,
            'verification': True,
            'verification_status': verification_status,
            'verification_type': scope.verification_type,
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
            'source_revision_changed': (
                self.tracker.source_revision != source_revision
            ),
            'available_validation_commands': [
                {
                    'id': item.id,
                    'command': item.command,
                    'target': item.target,
                    'source': item.source,
                }
                for item in discovered
            ],
        }
        content = render_process_output(result)
        finished_at = time()
        if self.ledger is not None:
            metadata['verification_ledger_recorded'] = True
            self.ledger.record_from_metadata(
                metadata,
                content=content,
                evidence_source='verify',
                reusable_key=key,
                started_at=started_at,
                finished_at=finished_at,
            )
        if result.timed_out:
            return ToolResult.fail(
                'verification_timeout',
                f'Verification timed out after '
                f'{arguments.timeout_seconds:g}s.',
                content=content,
                metadata=metadata,
            )
        if result.exit_code != 0:
            return ToolResult.fail(
                'verification_failed',
                f'Verification exited with code {result.exit_code}.',
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
        passed = ToolResult.ok(
            f'Verification passed in {result.duration_seconds:.3f}s.',
            content=content,
            metadata=metadata,
        )
        if scope.reusable:
            self.tracker.verification_cache[key] = passed
        return passed


def _current_fingerprints(
    tracker: WorkspaceTracker,
    paths: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, tracker.current.files[path])
        for path in paths
        if path in tracker.current.files
    )
