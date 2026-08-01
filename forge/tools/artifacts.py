'''Restricted runtime cleanup for trusted verification-created artifacts.'''

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from forge.runtime.workspace import WorkspaceTracker, fingerprint_path
from forge.tools.base import Tool, ToolInput, ToolResult

if TYPE_CHECKING:
    from forge.runtime.verification_ledger import VerificationLedger


class CleanupVerificationArtifactsInput(ToolInput):
    pass


class CleanupVerificationArtifactsTool(
    Tool[CleanupVerificationArtifactsInput]
):
    name = 'cleanup_verification_artifacts'
    description = (
        'Remove only files created by the latest successful verification for '
        'the current source revision. This runtime operation accepts no path '
        'arguments and never removes files that existed before verification.'
    )
    input_model = CleanupVerificationArtifactsInput
    effect = 'process'

    def __init__(
        self,
        root: Path,
        tracker: WorkspaceTracker,
        ledger: VerificationLedger | None = None,
    ) -> None:
        super().__init__(root)
        self.tracker = tracker
        self.ledger = ledger

    async def execute(
        self,
        arguments: CleanupVerificationArtifactsInput,
    ) -> ToolResult:
        del arguments
        if self.ledger is None:
            return ToolResult.fail(
                'artifact_cleanup_unavailable',
                'Verification ledger is unavailable.',
            )
        record = self.ledger.latest_for_source_revision(
            self.tracker.source_revision
        )
        if record is None or not record.success:
            return ToolResult.fail(
                'artifact_cleanup_unavailable',
                'No successful current-revision verification record exists.',
            )
        deleted: list[str] = []
        preserved: list[str] = []
        for delta in record.artifact_deltas:
            path = delta.path.replace('\\', '/')
            if delta.operation != 'created':
                preserved.append(path)
                continue
            if not _is_safe_relative_path(path):
                preserved.append(path)
                continue
            classified = self.tracker.classifier.classify_path(path)
            if classified.kind == 'forbidden':
                preserved.append(path)
                continue
            candidate = (self.root / Path(path)).resolve(strict=False)
            try:
                candidate.relative_to(self.root.resolve())
            except ValueError:
                preserved.append(path)
                continue
            if candidate.is_symlink() or not candidate.is_file():
                preserved.append(path)
                continue
            if fingerprint_path(self.root, path) != delta.after_fingerprint:
                preserved.append(path)
                continue
            try:
                candidate.unlink()
            except OSError:
                preserved.append(path)
                continue
            deleted.append(path)
        await self.tracker.refresh(origin='verification')
        return ToolResult.ok(
            f'Removed {len(deleted)} verification-created artifact(s).',
            metadata={
                'artifact_cleanup': True,
                'verification_record_command': record.command,
                'source_revision': self.tracker.source_revision,
                'deleted_paths': deleted,
                'preserved_paths': list(dict.fromkeys(preserved)),
            },
        )


def _is_safe_relative_path(path: str) -> bool:
    if not path or path.startswith('/') or '\\' in path:
        return False
    parts = PurePosixPath(path).parts
    return '..' not in parts and not any(':' in part for part in parts)
