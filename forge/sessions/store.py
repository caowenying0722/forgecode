
'''JSON persistence for resumable ForgeCode conversations.'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4

from forge.runtime.workspace import fingerprint_path, parse_porcelain_paths
from forge.sessions.rollout import (
    SessionRollout,
    message_delta_records,
    runtime_event_record,
)
from forge.tasks.state import ActiveTask


SESSION_ID_PATTERN = re.compile(r'session-[0-9a-f]{12}')


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    '''Model-visible state needed to continue a prior CLI conversation.'''

    id: str
    created_at: str
    updated_at: str
    cwd: str
    messages: list[dict[str, Any]]
    active_task: ActiveTask | None = None
    acceptance_ledger: dict[str, Any] | None = None
    interaction_mode: str = 'auto'
    permission_mode: str = 'trusted'
    git_head: str | None = None
    git_branch: str | None = None
    workspace_digest: str | None = None
    parent_session_id: str | None = None
    forked_at_seq: int | None = None
    resume_context: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'cwd': self.cwd,
            'messages': self.messages,
            'active_task': (
                self.active_task.as_dict()
                if self.active_task is not None
                else None
            ),
            'acceptance_ledger': self.acceptance_ledger,
            'interaction_mode': self.interaction_mode,
            'permission_mode': self.permission_mode,
            'git_head': self.git_head,
            'git_branch': self.git_branch,
            'workspace_digest': self.workspace_digest,
            'parent_session_id': self.parent_session_id,
            'forked_at_seq': self.forked_at_seq,
            'resume_context': self.resume_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        session_id = str(data['id'])
        validate_session_id(session_id)
        messages = data.get('messages', [])
        if not isinstance(messages, list):
            raise ValueError('Session messages must be a list.')
        active_task_data = data.get('active_task')
        return cls(
            id=session_id,
            created_at=str(data.get('created_at', '')),
            updated_at=str(data.get('updated_at', '')),
            cwd=str(data.get('cwd', '')),
            messages=[
                dict(message)
                for message in messages
                if isinstance(message, dict)
            ],
            active_task=(
                ActiveTask.from_dict(active_task_data)
                if isinstance(active_task_data, dict)
                else None
            ),
            acceptance_ledger=(
                dict(data['acceptance_ledger'])
                if isinstance(data.get('acceptance_ledger'), dict)
                else None
            ),
            interaction_mode=str(data.get('interaction_mode', 'auto')),
            permission_mode=str(data.get('permission_mode', 'trusted')),
            git_head=optional_string(data.get('git_head')),
            git_branch=optional_string(data.get('git_branch')),
            workspace_digest=optional_string(data.get('workspace_digest')),
            parent_session_id=optional_string(data.get('parent_session_id')),
            forked_at_seq=optional_int(data.get('forked_at_seq')),
            resume_context=(
                dict(data['resume_context'])
                if isinstance(data.get('resume_context'), dict)
                else None
            ),
        )


class SessionStore:
    '''Persist resumable sessions under the repository-local .forge folder.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'sessions'
        self.current_path = self.directory / 'current.json'
        self.rollout = SessionRollout(self.directory)

    def save(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        active_task: ActiveTask | None = None,
        interaction_mode: str = 'auto',
        permission_mode: str = 'trusted',
        update_workspace: bool = True,
        parent_session_id: str | None = None,
        forked_at_seq: int | None = None,
        resume_context: dict[str, Any] | None = None,
        acceptance_ledger: dict[str, Any] | None = None,
    ) -> SessionSnapshot:
        resolved_id = session_id or new_session_id()
        validate_session_id(resolved_id)
        existing_path = self.path_for(resolved_id)
        existing = self._read(existing_path) if existing_path.is_file() else None
        snapshot = self._build_snapshot(
            resolved_id,
            messages,
            existing=existing,
            active_task=active_task,
            interaction_mode=interaction_mode,
            permission_mode=permission_mode,
            update_workspace=update_workspace,
            parent_session_id=parent_session_id,
            forked_at_seq=forked_at_seq,
            resume_context=resume_context,
            acceptance_ledger=acceptance_ledger,
        )
        if existing is None:
            self._materialize(snapshot)
        rollout_existing = (
            existing
            if self.rollout.path_for(resolved_id).is_file()
            else None
        )
        records = self._snapshot_records(rollout_existing, snapshot)
        self.rollout.append(resolved_id, records)
        self._materialize(snapshot)
        return snapshot

    def record_event(
        self,
        event: object,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None,
        active_task: ActiveTask | None,
        interaction_mode: str,
        permission_mode: str,
        update_workspace: bool = False,
        resume_context: dict[str, Any] | None = None,
        acceptance_ledger: dict[str, Any] | None = None,
    ) -> SessionSnapshot:
        resolved_id = session_id or new_session_id()
        validate_session_id(resolved_id)
        existing_path = self.path_for(resolved_id)
        existing = self._read(existing_path) if existing_path.is_file() else None
        snapshot = self._build_snapshot(
            resolved_id,
            messages,
            existing=existing,
            active_task=active_task,
            interaction_mode=interaction_mode,
            permission_mode=permission_mode,
            update_workspace=update_workspace,
            resume_context=resume_context,
            acceptance_ledger=acceptance_ledger,
        )
        if existing is None:
            self._materialize(snapshot)
        rollout_existing = (
            existing
            if self.rollout.path_for(resolved_id).is_file()
            else None
        )
        snapshot_records = self._snapshot_records(
            rollout_existing,
            snapshot,
        )
        runtime_record = runtime_event_record(event)
        records = (
            [*snapshot_records, runtime_record]
            if rollout_existing is None
            else [runtime_record, *snapshot_records]
        )
        self.rollout.append(resolved_id, records)
        if existing is None or snapshot_records:
            self._materialize(snapshot)
        return snapshot

    def fork(self, session_id: str | None = None) -> SessionSnapshot:
        source = self.load(session_id) if session_id else self.load_current()
        child_id = new_session_id()
        return self.save(
            source.messages,
            session_id=child_id,
            active_task=source.active_task,
            interaction_mode=source.interaction_mode,
            permission_mode=source.permission_mode,
            parent_session_id=source.id,
            forked_at_seq=self.rollout.last_sequence(source.id),
            resume_context=source.resume_context,
            acceptance_ledger=source.acceptance_ledger,
        )

    def consistency_warnings(self, snapshot: SessionSnapshot) -> tuple[str, ...]:
        warnings: list[str] = []
        if Path(snapshot.cwd).resolve() != self.root:
            warnings.append(
                f'saved cwd is {snapshot.cwd}, current cwd is {self.root}'
            )
        current = workspace_identity(self.root)
        for label, saved, actual in (
            ('Git HEAD', snapshot.git_head, current['git_head']),
            ('Git branch', snapshot.git_branch, current['git_branch']),
            (
                'working tree',
                snapshot.workspace_digest,
                current['workspace_digest'],
            ),
        ):
            if saved is not None and actual is not None and saved != actual:
                warnings.append(f'{label} differs from the saved session')
        return tuple(warnings)

    def _build_snapshot(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        existing: SessionSnapshot | None,
        active_task: ActiveTask | None,
        interaction_mode: str,
        permission_mode: str,
        update_workspace: bool,
        parent_session_id: str | None = None,
        forked_at_seq: int | None = None,
        resume_context: dict[str, Any] | None = None,
        acceptance_ledger: dict[str, Any] | None = None,
    ) -> SessionSnapshot:
        now = datetime.now().astimezone().isoformat()
        identity = (
            workspace_identity(self.root)
            if update_workspace or existing is None
            else {
                'git_head': existing.git_head,
                'git_branch': existing.git_branch,
                'workspace_digest': existing.workspace_digest,
            }
        )
        return SessionSnapshot(
            id=session_id,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            cwd=str(self.root),
            messages=json_round_trip(messages),
            active_task=active_task,
            acceptance_ledger=(
                json_round_trip(acceptance_ledger)
                if acceptance_ledger is not None
                else existing.acceptance_ledger if existing is not None else None
            ),
            interaction_mode=interaction_mode,
            permission_mode=permission_mode,
            git_head=identity['git_head'],
            git_branch=identity['git_branch'],
            workspace_digest=identity['workspace_digest'],
            parent_session_id=(
                parent_session_id
                if parent_session_id is not None
                else existing.parent_session_id if existing else None
            ),
            forked_at_seq=(
                forked_at_seq
                if forked_at_seq is not None
                else existing.forked_at_seq if existing else None
            ),
            resume_context=(
                json_round_trip(resume_context)
                if resume_context is not None
                else existing.resume_context if existing else None
            ),
        )

    def _snapshot_records(
        self,
        existing: SessionSnapshot | None,
        snapshot: SessionSnapshot,
    ) -> list[tuple[str, dict[str, Any]]]:
        records: list[tuple[str, dict[str, Any]]] = []
        if existing is None:
            records.append(
                (
                    'session_started',
                    {
                        key: value
                        for key, value in snapshot.as_dict().items()
                        if key != 'messages'
                    },
                )
            )
        records.extend(
            message_delta_records(
                existing.messages if existing is not None else [],
                snapshot.messages,
                reference_tool_results=existing is not None,
            )
        )
        state = {
            key: value
            for key, value in snapshot.as_dict().items()
            if key != 'messages'
        }
        if existing is None or any(
            existing.as_dict().get(key) != value
            for key, value in state.items()
            if key not in {'updated_at'}
        ):
            records.append(('session_state', state))
        return records

    def _materialize(self, snapshot: SessionSnapshot) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            snapshot.as_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        self._write(self.path_for(snapshot.id), serialized)
        self._write(self.current_path, serialized)

    def load(self, session_id: str) -> SessionSnapshot:
        validate_session_id(session_id)
        path = self.path_for(session_id)
        fallback = self._read_dict(path) if path.is_file() else {'id': session_id}
        replayed = self.rollout.replay(session_id, fallback)
        return SessionSnapshot.from_dict(replayed)

    def load_current(self) -> SessionSnapshot:
        if not self.current_path.is_file():
            raise FileNotFoundError('No saved ForgeCode session exists.')
        current = self._read(self.current_path)
        return self.load(current.id)

    def list(self) -> tuple[SessionSnapshot, ...]:
        if not self.directory.exists():
            return ()
        snapshots: list[SessionSnapshot] = []
        for path in sorted(
            self.directory.glob('session-*.json'),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                snapshots.append(self.load(path.stem))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return tuple(snapshots)

    def exists(self, session_id: str) -> bool:
        return (
            self.path_for(session_id).is_file()
            or self.rollout.path_for(session_id).is_file()
        )

    def path_for(self, session_id: str) -> Path:
        validate_session_id(session_id)
        return self.directory / f'{session_id}.json'

    @staticmethod
    def _read(path: Path) -> SessionSnapshot:
        return SessionSnapshot.from_dict(SessionStore._read_dict(path))

    @staticmethod
    def _read_dict(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'Invalid session file: {path}')
        return data

    @staticmethod
    def _write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content + '\n', encoding='utf-8')
        temporary.replace(path)


def new_session_id() -> str:
    return f'session-{uuid4().hex[:12]}'


def validate_session_id(session_id: str) -> None:
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError(f'Invalid session ID: {session_id}')


def json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def workspace_identity(root: Path) -> dict[str, str | None]:
    def git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ['git', *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=10,
        )

    try:
        head_result = git('rev-parse', 'HEAD')
        branch_result = git('branch', '--show-current')
        status_result = git(
            'status',
            '--porcelain=v1',
            '-z',
            '--untracked-files=all',
            '--ignored=no',
        )
    except (OSError, subprocess.TimeoutExpired):
        return {'git_head': None, 'git_branch': None, 'workspace_digest': None}
    if head_result.returncode != 0:
        return {'git_head': None, 'git_branch': None, 'workspace_digest': None}
    digest: str | None = None
    if status_result.returncode == 0:
        hasher = sha256(status_result.stdout)
        status_text = status_result.stdout.decode('utf-8', errors='replace')
        for path in parse_porcelain_paths(status_text):
            hasher.update(path.encode('utf-8', errors='replace'))
            hasher.update(fingerprint_path(root, path).encode('utf-8'))
        digest = hasher.hexdigest()
    return {
        'git_head': head_result.stdout.decode().strip() or None,
        'git_branch': branch_result.stdout.decode().strip() or None,
        'workspace_digest': digest,
    }


def optional_string(value: Any) -> str | None:
    return str(value) if value not in {None, ''} else None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
