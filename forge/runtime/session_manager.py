'''Runtime-facing session persistence role.'''

from __future__ import annotations

from forge.sessions.store import SessionSnapshot, SessionStore
from forge.tasks.state import ActiveTask


class SessionManager:
    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def save(
        self,
        messages: list[dict[str, object]],
        *,
        session_id: str | None,
        active_task: ActiveTask | None,
        interaction_mode: str,
        permission_mode: str,
    ) -> SessionSnapshot:
        return self.store.save(
            messages,
            session_id=session_id,
            active_task=active_task,
            interaction_mode=interaction_mode,
            permission_mode=permission_mode,
        )

    def load(self, session_id: str | None) -> SessionSnapshot:
        return (
            self.store.load(session_id)
            if session_id is not None
            else self.store.load_current()
        )

    def history(self) -> str:
        sessions = self.store.list()
        if not sessions:
            return 'No saved sessions.'
        lines: list[str] = []
        for snapshot in sessions[:20]:
            task = snapshot.active_task.goal if snapshot.active_task else ''
            suffix = f' — {task[:80]}' if task else ''
            lines.append(
                f'- {snapshot.id} [{len(snapshot.messages)} messages] '
                f'{snapshot.updated_at}{suffix}'
            )
        return '\n'.join(lines)
