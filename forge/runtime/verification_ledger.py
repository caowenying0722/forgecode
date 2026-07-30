'''Turn-scoped verification ledger shared by completion and recovery.'''

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import time
from typing import Literal

from forge.runtime.state import VerificationEvidence


VerificationEvidenceSource = Literal['verify', 'run_command', 'cache']


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    target: str
    kind: str
    command: str
    cwd: str
    status: str
    exit_code: int
    source_revision: int
    filesystem_revision: int
    failure_signature: str
    artifact_scope: tuple[dict[str, object], ...] = ()
    side_effect_paths: tuple[str, ...] = ()
    generated_artifact_paths: tuple[str, ...] = ()
    cache_paths: tuple[str, ...] = ()
    generated_artifact_fingerprints: tuple[tuple[str, str], ...] = ()
    cache_fingerprints: tuple[tuple[str, str], ...] = ()
    stdout_stderr_summary: str = ''
    started_at: float = 0.0
    finished_at: float = 0.0
    evidence_source: VerificationEvidenceSource = 'verify'
    reusable_key: tuple[object, ...] = ()
    duration_seconds: float = 0.0
    timed_out: bool = False
    command_id: str = ''
    reused: bool = False

    @property
    def success(self) -> bool:
        return self.status == 'passed' and self.exit_code == 0 and not self.timed_out

    def to_evidence(self) -> VerificationEvidence:
        return VerificationEvidence(
            command=self.command,
            cwd=self.cwd,
            exit_code=self.exit_code,
            duration_seconds=self.duration_seconds,
            timed_out=self.timed_out,
            workspace_revision=self.source_revision,
            source_revision=self.source_revision,
            filesystem_revision=self.filesystem_revision,
            status=self.status,
            command_id=self.command_id,
            failure_signature=self.failure_signature,
            verification_type=self.kind,
            verification_reused=self.reused,
            verification_side_effect_paths=self.side_effect_paths,
            generated_artifact_paths=self.generated_artifact_paths,
            cache_paths=self.cache_paths,
            generated_artifact_fingerprints=self.generated_artifact_fingerprints,
            cache_fingerprints=self.cache_fingerprints,
        )


@dataclass(slots=True)
class VerificationLedger:
    '''Authoritative turn-local verification records keyed by source revision.'''

    records: list[VerificationRecord] = field(default_factory=list)
    reusable_successes: dict[tuple[object, ...], VerificationRecord] = field(
        default_factory=dict
    )

    def clear_turn(self) -> None:
        self.records.clear()

    def record(self, record: VerificationRecord) -> VerificationRecord:
        self.records.append(record)
        if record.success and record.reusable_key:
            self.reusable_successes[record.reusable_key] = record
        return record

    def record_from_metadata(
        self,
        metadata: dict[str, object],
        *,
        content: str = '',
        evidence_source: VerificationEvidenceSource = 'verify',
        reusable_key: tuple[object, ...] = (),
        started_at: float | None = None,
        finished_at: float | None = None,
    ) -> VerificationRecord | None:
        if metadata.get('verification') is not True:
            return None
        try:
            status = str(metadata.get('verification_status', ''))
            command = str(metadata['command'])
            cwd = str(metadata.get('cwd', '.'))
            source_revision = int(
                metadata.get('source_revision', metadata['workspace_revision'])
            )
            filesystem_revision = int(metadata.get('filesystem_revision', 0))
            exit_code = int(metadata.get('exit_code', -1))
            duration_seconds = float(metadata.get('duration_seconds', 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        now = time()
        failure_signature = str(metadata.get('failure_signature', ''))
        if not failure_signature and status != 'passed':
            failure_signature = _failure_signature(metadata, content)
        record = VerificationRecord(
            target=str(metadata.get('command_id', 'custom')),
            kind=str(metadata.get('verification_type', 'auto')),
            command=command,
            cwd=cwd,
            status=status,
            exit_code=exit_code,
            source_revision=source_revision,
            filesystem_revision=filesystem_revision,
            failure_signature=failure_signature,
            artifact_scope=tuple(
                item
                for item in metadata.get('verification_artifact_scope', [])
                if isinstance(item, dict)
            ),
            side_effect_paths=tuple(
                str(path)
                for path in metadata.get('verification_side_effect_paths', [])
            ),
            generated_artifact_paths=tuple(
                str(path)
                for path in metadata.get('generated_artifact_paths', [])
            ),
            cache_paths=tuple(
                str(path) for path in metadata.get('cache_paths', [])
            ),
            generated_artifact_fingerprints=_fingerprints_from_metadata(
                metadata.get('generated_artifact_fingerprints', [])
            ),
            cache_fingerprints=_fingerprints_from_metadata(
                metadata.get('cache_fingerprints', [])
            ),
            stdout_stderr_summary=content[:4_000],
            started_at=float(started_at if started_at is not None else now),
            finished_at=float(finished_at if finished_at is not None else now),
            evidence_source=evidence_source,
            reusable_key=reusable_key,
            duration_seconds=duration_seconds,
            timed_out=bool(metadata.get('timed_out', False)),
            command_id=str(metadata.get('command_id', '')),
            reused=bool(metadata.get('verification_reused', False)),
        )
        return self.record(record)

    def latest_for_source_revision(
        self,
        source_revision: int,
    ) -> VerificationRecord | None:
        return next(
            (
                record
                for record in reversed(self.records)
                if record.source_revision == source_revision
            ),
            None,
        )

    def latest_evidence(
        self,
        source_revision: int | None = None,
    ) -> VerificationEvidence | None:
        if source_revision is None:
            record = self.records[-1] if self.records else None
        else:
            record = self.latest_for_source_revision(source_revision)
        return record.to_evidence() if record is not None else None

    def reusable(self, key: tuple[object, ...]) -> VerificationRecord | None:
        return self.reusable_successes.get(key)


def _failure_signature(metadata: dict[str, object], content: str) -> str:
    signature_text = '\n'.join(
        str(metadata.get(key, ''))
        for key in ('command', 'exit_code', 'stderr')
    )
    if not signature_text.strip():
        signature_text = content[:4_000]
    return hashlib.sha256(signature_text.encode('utf-8')).hexdigest()


def _fingerprints_from_metadata(value: object) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for item in value:
        if isinstance(item, dict):
            path = item.get('path')
            fingerprint = item.get('fingerprint')
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            path, fingerprint = item
        else:
            continue
        if path is None or fingerprint is None:
            continue
        pairs.append((str(path), str(fingerprint)))
    return tuple(pairs)
