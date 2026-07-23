
'''JSON persistence for resumable ForgeCode conversations.'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

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
    interaction_mode: str = 'auto'

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
            'interaction_mode': self.interaction_mode,
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
            interaction_mode=str(data.get('interaction_mode', 'auto')),
        )


class SessionStore:
    '''Persist resumable sessions under the repository-local .forge folder.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'sessions'
        self.current_path = self.directory / 'current.json'

    def save(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        active_task: ActiveTask | None = None,
        interaction_mode: str = 'auto',
    ) -> SessionSnapshot:
        resolved_id = session_id or new_session_id()
        validate_session_id(resolved_id)
        existing = self.load(resolved_id) if self.exists(resolved_id) else None
        now = datetime.now().astimezone().isoformat()
        snapshot = SessionSnapshot(
            id=resolved_id,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            cwd=str(self.root),
            messages=json_round_trip(messages),
            active_task=active_task,
            interaction_mode=interaction_mode,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            snapshot.as_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        self._write(self.path_for(resolved_id), serialized)
        self._write(self.current_path, serialized)
        return snapshot

    def load(self, session_id: str) -> SessionSnapshot:
        validate_session_id(session_id)
        return self._read(self.path_for(session_id))

    def load_current(self) -> SessionSnapshot:
        if not self.current_path.is_file():
            raise FileNotFoundError('No saved ForgeCode session exists.')
        return self._read(self.current_path)

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
                snapshots.append(self._read(path))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return tuple(snapshots)

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).is_file()

    def path_for(self, session_id: str) -> Path:
        validate_session_id(session_id)
        return self.directory / f'{session_id}.json'

    @staticmethod
    def _read(path: Path) -> SessionSnapshot:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'Invalid session file: {path}')
        return SessionSnapshot.from_dict(data)

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


def json_round_trip(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(messages, ensure_ascii=False, default=str))
