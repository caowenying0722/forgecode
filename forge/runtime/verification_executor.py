'''Single authoritative execution path for formal verification commands.'''

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from forge.runtime.process import (
    process_metadata,
    render_process_output,
    run_process,
)
from forge.runtime.verification import (
    NON_INTERACTIVE_ENV,
    ProjectCommandResolutionError,
    ResolvedProjectCommand,
    ValidationCommand,
    ValidationTarget,
    VerificationTransaction,
    resolve_project_command,
    verification_artifact_scope,
    verification_cache_key,
)
from forge.runtime.workspace import WorkspaceTracker
from forge.runtime.workspace_classification import (
    ArtifactDelta,
    ArtifactRule,
    VerificationArtifactScope,
    artifact_deltas_from_metadata,
    artifact_rule_for,
)
from forge.tools.base import ToolResult

if TYPE_CHECKING:
    from forge.runtime.verification_ledger import (
        VerificationEvidenceSource,
        VerificationLedger,
    )


class VerificationExecutor:
    '''Execute, classify, cache, and record one formal verification.'''

    def __init__(
        self,
        root: Path,
        tracker: WorkspaceTracker,
        ledger: VerificationLedger | None = None,
    ) -> None:
        self.root = root.resolve()
        self.tracker = tracker
        self.ledger = ledger

    async def execute(
        self,
        *,
        command: str,
        cwd: Path,
        display_cwd: str,
        timeout_seconds: float,
        command_id: str,
        target: ValidationTarget,
        evidence_source: VerificationEvidenceSource,
        discovered_commands: tuple[ValidationCommand, ...] = (),
    ) -> ToolResult:
        source_revision = self.tracker.source_revision
        filesystem_revision = self.tracker.filesystem_revision
        try:
            resolved = resolve_project_command(command, cwd)
        except ProjectCommandResolutionError as error:
            return ToolResult.fail(
                error.code,
                str(error),
                metadata={
                    'verification': True,
                    'verification_status': 'invalid',
                    'command': command,
                    'command_id': command_id,
                    'cwd': display_cwd,
                    'workspace_revision': source_revision,
                    'source_revision': source_revision,
                    'filesystem_revision': filesystem_revision,
                    'exit_code': -1,
                    'duration_seconds': 0.0,
                    'timed_out': False,
                    'stderr': str(error),
                    'resolution_error': {
                        'code': error.code,
                        'path': str(error.path) if error.path is not None else '',
                        'script_chain': list(error.script_chain),
                    },
                },
            )
        scope = resolved.artifact_scope
        if resolved.effective_commands == (command.strip(),):
            scope = verification_artifact_scope(
                command,
                root=cwd,
                target=target,
            )
            resolved = ResolvedProjectCommand(
                invoked_command=resolved.invoked_command,
                effective_commands=resolved.effective_commands,
                verification_types=(
                    resolved.verification_types
                    or (
                        (scope.verification_type,)
                        if scope.verification_type != 'auto'
                        else ()
                    )
                ),
                artifact_scope=scope,
                script_chain=resolved.script_chain,
            )
        workspace_scope = _workspace_relative_scope(
            scope,
            workspace_root=self.root,
            command_cwd=cwd,
        )
        key = verification_cache_key(
            source_revision=source_revision,
            command=command,
            cwd=display_cwd,
            scope=workspace_scope,
            resolved_commands=resolved.effective_commands,
            dependency_fingerprint=_dependency_manifest_fingerprint(cwd),
        )
        cached = self.tracker.verification_cache.get(key)
        if isinstance(cached, ToolResult) and _cached_artifacts_match(
            self.tracker,
            cached,
        ):
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
        if cached is not None:
            self.tracker.verification_cache.pop(key, None)

        before_snapshot = await self.tracker.capture_transaction_snapshot(
            workspace_scope
        )
        if before_snapshot is None:
            return ToolResult.fail(
                'workspace_snapshot_unavailable',
                'Verification was not started because its before snapshot '
                'could not be captured.',
                metadata={
                    'verification': True,
                    'verification_status': 'failed',
                    'command': command,
                    'command_id': command_id,
                    'cwd': display_cwd,
                    'workspace_revision': source_revision,
                    'source_revision': source_revision,
                    'filesystem_revision': filesystem_revision,
                    'exit_code': -1,
                    'duration_seconds': 0.0,
                    'timed_out': False,
                },
            )
        started_at = time()
        process = await run_process(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            shell=True,
            env=NON_INTERACTIVE_ENV,
        )
        change = await self.tracker.refresh(
            origin='verification',
            artifact_scope=workspace_scope,
            before_snapshot=before_snapshot,
        )
        if change is None:
            return ToolResult.fail(
                'workspace_snapshot_unavailable',
                'Verification completed, but the workspace delta could not '
                'be captured.',
                content=render_process_output(process),
                metadata={
                    **process_metadata(process),
                    'verification': True,
                    'verification_status': 'failed',
                    'command': command,
                    'command_id': command_id,
                    'cwd': display_cwd,
                    'workspace_revision': source_revision,
                    'source_revision': source_revision,
                    'filesystem_revision': filesystem_revision,
                },
            )
        transaction = VerificationTransaction.from_workspace_change(
            command=command,
            cwd=display_cwd,
            source_revision_before=source_revision,
            filesystem_revision_before=filesystem_revision,
            change=change,
        )
        transaction = replace(
            transaction,
            artifact_deltas=_artifact_deltas(transaction, workspace_scope),
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
        generated_fingerprints = _current_fingerprints(
            self.tracker,
            classification.generated_paths,
        )
        cache_fingerprints = _current_fingerprints(
            self.tracker,
            classification.cache_paths,
        )
        metadata = {
            **process_metadata(process),
            'command': command,
            'invoked_command': resolved.invoked_command,
            'effective_commands': list(resolved.effective_commands),
            'verification_types': list(resolved.verification_types),
            'script_chain': list(resolved.script_chain),
            'command_id': command_id,
            'cwd': display_cwd,
            'workspace_revision': source_revision,
            'source_revision': source_revision,
            'filesystem_revision': self.tracker.filesystem_revision,
            'verification': True,
            'verification_status': verification_status,
            'verification_type': workspace_scope.verification_type,
            'verification_levels': list(
                verification_levels_for(
                    resolved.verification_types,
                    command=command,
                )
            ),
            'verification_artifact_scope': [
                {
                    'pattern': rule.pattern,
                    'kind': rule.kind,
                    'description': rule.description,
                }
                for rule in workspace_scope.allowed_writes
            ],
            'verification_side_effect_paths': list(side_effect_paths),
            'generated_artifact_paths': list(classification.generated_paths),
            'cache_paths': list(classification.cache_paths),
            'generated_artifact_fingerprints': [
                list(item) for item in generated_fingerprints
            ],
            'cache_fingerprints': [list(item) for item in cache_fingerprints],
            'verification_transaction': transaction.as_metadata(),
            'artifact_deltas': [
                delta.as_dict() for delta in transaction.artifact_deltas
            ],
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
                for item in discovered_commands
            ],
        }
        content = render_process_output(process)
        if self.ledger is not None:
            metadata['verification_ledger_recorded'] = True
            self.ledger.record_from_metadata(
                metadata,
                content=content,
                evidence_source=evidence_source,
                reusable_key=key,
                started_at=started_at,
                finished_at=time(),
            )
        if process.timed_out:
            return ToolResult.fail(
                'verification_timeout',
                f'Verification timed out after {timeout_seconds:g}s.',
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
        passed = ToolResult.ok(
            f'Verification passed in {process.duration_seconds:.3f}s.',
            content=content,
            metadata=metadata,
        )
        if workspace_scope.reusable:
            self.tracker.verification_cache[key] = passed
        return passed


def verification_levels_for(
    verification_types: tuple[str, ...],
    *,
    command: str,
) -> tuple[str, ...]:
    '''Map executed validators to claims no stronger than their evidence.'''
    mapping = {
        'typecheck': 'typecheck_verified',
        'test': 'unit_tests_verified',
        'build': 'build_verified',
    }
    levels = [
        mapping[item]
        for item in verification_types
        if item in mapping
    ]
    if command.strip().casefold().startswith('git diff'):
        levels.append('diff_verified')
    return tuple(dict.fromkeys(levels))


def _workspace_relative_scope(
    scope: VerificationArtifactScope,
    *,
    workspace_root: Path,
    command_cwd: Path,
) -> VerificationArtifactScope:
    relative = command_cwd.resolve().relative_to(workspace_root.resolve())
    prefix = relative.as_posix().strip('.')
    if not prefix:
        return scope

    def prefixed(pattern: str) -> str:
        return f'{prefix}/{pattern.lstrip("/")}'

    return VerificationArtifactScope(
        verification_type=scope.verification_type,
        read_patterns=tuple(prefixed(item) for item in scope.read_patterns),
        allowed_writes=tuple(
            ArtifactRule(
                prefixed(rule.pattern),
                rule.kind,
                rule.description,
            )
            for rule in scope.allowed_writes
        ),
        forbidden_source_patterns=tuple(
            prefixed(item) for item in scope.forbidden_source_patterns
        ),
        allow_network=scope.allow_network,
        allow_dependency_install=scope.allow_dependency_install,
        cleanup_generated=scope.cleanup_generated,
        reusable=scope.reusable,
    )


def _current_fingerprints(
    tracker: WorkspaceTracker,
    paths: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, tracker.current.files[path])
        for path in paths
        if path in tracker.current.files
    )


def _artifact_deltas(
    transaction: VerificationTransaction,
    scope: VerificationArtifactScope,
) -> tuple[ArtifactDelta, ...]:
    before = dict(transaction.before_fingerprints)
    after = dict(transaction.after_fingerprints)
    operations = {
        **{path: 'created' for path in transaction.created_paths},
        **{path: 'modified' for path in transaction.modified_paths},
        **{path: 'deleted' for path in transaction.deleted_paths},
    }
    deltas: list[ArtifactDelta] = []
    for change in transaction.classification.changes:
        if change.kind not in {'generated_artifact', 'cache'}:
            continue
        rule = artifact_rule_for(change.path, scope)
        operation = operations.get(change.path)
        if rule is None or operation is None:
            continue
        before_fingerprint = before.get(change.path, 'missing')
        after_fingerprint = after.get(change.path, 'missing')
        deltas.append(
            ArtifactDelta(
                path=change.path,
                operation=operation,  # type: ignore[arg-type]
                kind=change.kind,  # type: ignore[arg-type]
                before_fingerprint=(
                    None if before_fingerprint == 'missing' else before_fingerprint
                ),
                after_fingerprint=(
                    None if after_fingerprint == 'missing' else after_fingerprint
                ),
                rule_pattern=rule.pattern,
                rule_reason=rule.description,
            )
        )
    return tuple(deltas)


def _cached_artifacts_match(
    tracker: WorkspaceTracker,
    cached: ToolResult,
) -> bool:
    metadata = cached.metadata
    deltas = artifact_deltas_from_metadata(metadata.get('artifact_deltas', []))
    for delta in deltas:
        current = _path_fingerprint(tracker.root, delta.path)
        if delta.operation == 'deleted':
            if current != 'missing':
                return False
        elif current != delta.after_fingerprint:
            return False
    fingerprint_pairs = (
        *metadata.get('generated_artifact_fingerprints', []),
        *metadata.get('cache_fingerprints', []),
    )
    checked_paths: set[str] = {delta.path for delta in deltas}
    for item in fingerprint_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        path, expected = str(item[0]), str(item[1])
        if path in checked_paths:
            continue
        if _path_fingerprint(tracker.root, path) != expected:
            return False
        checked_paths.add(path)
    declared = {
        str(path)
        for path in (
            *metadata.get('generated_artifact_paths', []),
            *metadata.get('cache_paths', []),
        )
    }
    return declared <= checked_paths


def _path_fingerprint(root: Path, path: str) -> str:
    from forge.runtime.workspace import fingerprint_path

    return fingerprint_path(root, path)


def _dependency_manifest_fingerprint(root: Path) -> str:
    names = (
        'package.json',
        'package-lock.json',
        'pnpm-lock.yaml',
        'yarn.lock',
        'bun.lock',
        'bun.lockb',
        'pyproject.toml',
        'uv.lock',
        'Cargo.toml',
        'Cargo.lock',
        'go.mod',
        'go.sum',
    )
    digest = sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        try:
            digest.update(path.read_bytes())
        except OSError as error:
            digest.update(f'unreadable:{error.errno}'.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()
