'''Tests for resumable session persistence.'''

from pathlib import Path

from forge.sessions.store import SessionStore
from forge.tasks.state import ActiveTask


def test_session_store_saves_current_and_lists_latest(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    task = ActiveTask(id='task-123456789abc', goal='Fix bug')
    snapshot = store.save(
        [{'role': 'user', 'content': 'hello'}],
        active_task=task,
        interaction_mode='plan',
    )

    current = store.load_current()
    listed = store.list()

    assert current.id == snapshot.id
    assert current.messages == [{'role': 'user', 'content': 'hello'}]
    assert current.active_task is not None
    assert current.active_task.goal == 'Fix bug'
    assert current.interaction_mode == 'plan'
    assert listed[0].id == snapshot.id


def test_session_store_reuses_session_id_on_save(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    first = store.save([{'role': 'user', 'content': 'hello'}])
    second = store.save(
        [
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi'},
        ],
        session_id=first.id,
    )

    assert second.id == first.id
    assert second.created_at == first.created_at
    assert len(store.load(first.id).messages) == 2
