'''Tests for team message bus tools.'''

from __future__ import annotations

import asyncio
from pathlib import Path

from forge.runtime.team import MessageBus, render_team_notification
from forge.tasks.graph import TaskGraphStore
from forge.tools.team import (
    CheckInboxTool,
    ClaimNextTaskTool,
    ListTeamRequestsTool,
    RequestTeamActionTool,
    RespondTeamRequestTool,
    SendMessageTool,
)


def run(coroutine: object):
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_send_message_and_check_inbox_consumes_messages(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    send = SendMessageTool(tmp_path, bus=bus, sender='task_subagent')
    check = CheckInboxTool(tmp_path, bus=bus, recipient='lead')

    sent = run(
        send.run(
            {
                'to': 'lead',
                'type': 'status',
                'content': 'Found the relevant files.',
            }
        )
    )
    collected = run(check.run({}))
    empty = run(check.run({}))

    assert sent.success is True
    assert sent.metadata['from'] == 'task_subagent'
    assert collected.success is True
    assert collected.metadata['message_count'] == 1
    assert '<team_message>' in collected.content
    assert '<from>task_subagent</from>' in collected.content
    assert 'Found the relevant files.' in collected.content
    assert empty.success is True
    assert empty.metadata['message_count'] == 0


def test_team_notification_escapes_xml_sensitive_content(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    message = bus.send(
        sender='lead',
        recipient='task_subagent',
        message_type='warning',
        content='Use A < B & C > D',
    )

    rendered = render_team_notification((message,))[0]

    assert 'A &lt; B &amp; C &gt; D' in rendered


def test_team_request_response_tracks_protocol_state(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    request_tool = RequestTeamActionTool(tmp_path, bus=bus, sender='lead')
    respond_tool = RespondTeamRequestTool(
        tmp_path,
        bus=bus,
        sender='task_subagent',
    )
    list_tool = ListTeamRequestsTool(tmp_path, bus=bus)

    requested = run(
        request_tool.run(
            {
                'to': 'task_subagent',
                'type': 'shutdown',
                'content': 'Please shut down gracefully.',
            }
        )
    )
    inbox = bus.collect('task_subagent')
    responded = run(
        respond_tool.run(
            {
                'request_id': requested.metadata['request_id'],
                'approve': True,
                'content': 'Shutdown approved.',
                'reply_to': inbox[0].id,
            }
        )
    )
    listed = run(list_tool.run({'status': 'approved'}))
    lead_messages = bus.collect('lead')

    assert requested.success is True
    assert inbox[0].type == 'shutdown_request'
    assert inbox[0].request_id == requested.metadata['request_id']
    assert responded.success is True
    assert responded.metadata['status'] == 'approved'
    assert lead_messages[0].type == 'shutdown_response'
    assert lead_messages[0].approve is True
    assert lead_messages[0].reply_to == inbox[0].id
    assert requested.metadata['request_id'] in listed.content


def test_team_response_rejects_wrong_target(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    request, _ = bus.request(
        sender='lead',
        recipient='alice',
        request_type='plan_approval',
        content='Review this plan.',
    )
    respond_tool = RespondTeamRequestTool(tmp_path, bus=bus, sender='bob')

    result = run(
        respond_tool.run(
            {
                'request_id': request.request_id,
                'approve': False,
                'content': 'Not mine.',
            }
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'team_response_rejected'
    assert bus.load_request(request.request_id).status == 'pending'


def test_claim_next_task_claims_first_unblocked_task(tmp_path: Path) -> None:
    store = TaskGraphStore(tmp_path)
    first = store.create('Prepare schema')
    blocked = store.create('Implement API', blocked_by=[first.id])
    available = store.create('Write docs')
    tool = ClaimNextTaskTool(
        tmp_path,
        store=store,
        agent_id='task_subagent',
    )

    result = run(tool.run({}))

    assert result.success is True
    assert result.metadata['task_id'] in {first.id, available.id}
    assert store.load(result.metadata['task_id']).owner == 'task_subagent'
    assert store.load(blocked.id).status == 'pending'
