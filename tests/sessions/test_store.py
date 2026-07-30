'''Tests for resumable session persistence.'''

from pathlib import Path

from forge.runtime.state import (
    ModelCallStarted,
    ToolCall,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from forge.runtime.session_manager import SessionManager
from forge.sessions.store import SessionStore
from forge.tasks.state import ActiveTask
from forge.tools.base import ToolResult


def test_session_store_saves_current_and_lists_latest(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    task = ActiveTask(id='task-123456789abc', goal='Fix bug')
    snapshot = store.save(
        [{'role': 'user', 'content': 'hello'}],
        active_task=task,
        interaction_mode='plan',
        permission_mode='readonly',
    )

    current = store.load_current()
    listed = store.list()

    assert current.id == snapshot.id
    assert current.messages == [{'role': 'user', 'content': 'hello'}]
    assert current.active_task is not None
    assert current.active_task.goal == 'Fix bug'
    assert current.interaction_mode == 'plan'
    assert current.permission_mode == 'readonly'
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


def test_session_store_saves_full_history(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    messages = [
        message
        for index in range(16)
        for message in (
            {'role': 'user', 'content': f'user {index}'},
            {'role': 'assistant', 'content': f'assistant {index}'},
        )
    ]

    snapshot = store.save(messages)
    resumed = store.load(snapshot.id)

    assert resumed.messages[0] == {'role': 'user', 'content': 'user 0'}
    assert resumed.messages[-1] == {
        'role': 'assistant',
        'content': 'assistant 15',
    }
    assert [
        message['content']
        for message in resumed.messages
        if message.get('role') == 'user'
    ] == [f'user {index}' for index in range(16)]


def test_session_manager_choices_default_to_latest_15_sessions(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store)
    saved_ids = [
        store.save([{'role': 'user', 'content': f'user {index}'}]).id
        for index in range(17)
    ]

    choices = manager.choices()

    assert len(choices) == 15
    assert [choice[0] for choice in choices] == list(reversed(saved_ids[-15:]))


def test_rollout_repairs_completed_tool_after_interrupted_batch(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    user = {'role': 'user', 'content': 'change the file'}
    call = ToolCall(
        index=0,
        id='tool-write',
        name='write_file',
        arguments={'path': 'a.txt', 'content': 'done'},
    )
    assistant = {
        'role': 'assistant',
        'content': [
            {
                'type': 'tool_use',
                'id': call.id,
                'name': call.name,
                'input': call.arguments,
            }
        ],
    }
    started = store.record_event(
        ModelCallStarted(iteration=1),
        [user],
        session_id=None,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )
    store.record_event(
        ToolExecutionStarted(tool_call=call),
        [user, assistant],
        session_id=started.id,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )
    store.record_event(
        ToolExecutionCompleted(
            tool_call=call,
            result=ToolResult.ok('Wrote a.txt.', content='persisted result'),
        ),
        [user, assistant],
        session_id=started.id,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )

    resumed = store.load(started.id)

    assert resumed.messages[:2] == [user, assistant]
    tool_result = resumed.messages[-1]['content'][0]
    assert tool_result['tool_use_id'] == call.id
    assert tool_result['is_error'] is False
    assert 'persisted result' in tool_result['content']
    records = store.rollout.read(started.id)
    assert records[-1]['type'] == 'tool_execution_completed'

    store.save(resumed.messages, session_id=started.id)
    committed = store.load(started.id)
    committed_records = store.rollout.read(started.id)
    encoded = committed_records[-1]['payload']['messages'][0]
    assert encoded == {'$tool_results': [call.id]}
    assert committed.messages == resumed.messages


def test_session_store_forks_without_mutating_parent(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    parent = store.save([{'role': 'user', 'content': 'original'}])

    child = store.fork(parent.id)

    assert child.id != parent.id
    assert child.parent_session_id == parent.id
    assert child.forked_at_seq == store.rollout.last_sequence(parent.id)
    assert child.messages == parent.messages
    assert store.load(parent.id).parent_session_id is None


def test_rollout_marks_unfinished_tools_as_interrupted(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    call = ToolCall(0, 'tool-read', 'read_file', {'path': 'a.txt'})
    pending = ToolCall(1, 'tool-write', 'write_file', {'path': 'b.txt'})
    messages = [
        {'role': 'user', 'content': 'inspect and change'},
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': item.id,
                    'name': item.name,
                    'input': item.arguments,
                }
                for item in (call, pending)
            ],
        },
    ]
    snapshot = store.record_event(
        ToolExecutionStarted(tool_call=call),
        messages,
        session_id=None,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )
    store.record_event(
        ToolExecutionCompleted(
            tool_call=call,
            result=ToolResult.ok('Read a.txt.'),
        ),
        messages,
        session_id=snapshot.id,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )

    resumed = store.load(snapshot.id)
    results = resumed.messages[-1]['content']

    assert results[0]['is_error'] is False
    assert results[1]['is_error'] is True
    assert 'interrupted_tool_call' in results[1]['content']


def test_rollout_ignores_one_truncated_trailing_record(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    snapshot = store.save([{'role': 'user', 'content': 'safe'}])
    with store.rollout.path_for(snapshot.id).open('a', encoding='utf-8') as file:
        file.write('{"seq":999,"type":"partial"')

    resumed = store.load(snapshot.id)

    assert resumed.messages == snapshot.messages


def test_first_rollout_event_bootstraps_legacy_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    legacy = store.save([{'role': 'user', 'content': 'legacy'}])
    store.rollout.path_for(legacy.id).unlink()

    store.record_event(
        ModelCallStarted(iteration=1),
        [
            {'role': 'user', 'content': 'legacy'},
            {'role': 'assistant', 'content': 'continued'},
        ],
        session_id=legacy.id,
        active_task=None,
        interaction_mode='auto',
        permission_mode='trusted',
    )

    resumed = store.load(legacy.id)
    assert [message['content'] for message in resumed.messages] == [
        'legacy',
        'continued',
    ]
