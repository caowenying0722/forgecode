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

    def record_event(
        self,
        event: object,
        messages: list[dict[str, object]],
        *,
        session_id: str | None,
        active_task: ActiveTask | None,
        interaction_mode: str,
        permission_mode: str,
        update_workspace: bool = False,
    ) -> SessionSnapshot:
        return self.store.record_event(
            event,
            messages,
            session_id=session_id,
            active_task=active_task,
            interaction_mode=interaction_mode,
            permission_mode=permission_mode,
            update_workspace=update_workspace,
        )

    def load(self, session_id: str | None) -> SessionSnapshot:
        return (
            self.store.load(session_id)
            if session_id is not None
            else self.store.load_current()
        )

    def fork(self, session_id: str | None = None) -> SessionSnapshot:
        return self.store.fork(session_id)

    def consistency_warnings(
        self,
        snapshot: SessionSnapshot,
    ) -> tuple[str, ...]:
        return self.store.consistency_warnings(snapshot)

    def history(self) -> str:
        sessions = self.store.list()
        if not sessions:
            return 'No saved sessions.'
        lines: list[str] = []
        for snapshot in sessions[:20]:
            task = snapshot.active_task.goal if snapshot.active_task else ''
            suffix = f' — {task[:80]}' if task else ''
            parent = (
                f' forked from {snapshot.parent_session_id}'
                if snapshot.parent_session_id
                else ''
            )
            lines.append(
                f'- {snapshot.id} [{len(snapshot.messages)} messages] '
                f'{snapshot.updated_at}{parent}{suffix}'
            )
        return '\n'.join(lines)

    def choices(self, limit: int | None = 15) -> tuple[tuple[str, str, str], ...]:
        '''Return saved sessions as terminal picker choices.'''
        choices: list[tuple[str, str, str]] = []
        sessions = self.store.list()
        if limit is not None:
            sessions = sessions[:limit]
        for snapshot in sessions:
            task = snapshot.active_task.goal if snapshot.active_task else ''
            label = f'{snapshot.id}  {snapshot.updated_at}'
            details = [f'{len(snapshot.messages)} message(s)']
            if snapshot.interaction_mode:
                details.append(f'mode {snapshot.interaction_mode}')
            if snapshot.parent_session_id:
                details.append(f'forked from {snapshot.parent_session_id}')
            if task:
                details.append(task[:80])
            choices.append((snapshot.id, label, ' · '.join(details)))
        return tuple(choices)
