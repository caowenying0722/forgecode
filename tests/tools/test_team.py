'''Tests for team message bus tools.'''

from __future__ import annotations

import asyncio
from pathlib import Path

from forge.runtime.team import MessageBus, render_team_notification
from forge.tools.team import CheckInboxTool, SendMessageTool


def run(coroutine: object):
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_send_message_and_check_inbox_consumes_messages(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    send = SendMessageTool(tmp_path, bus=bus, sender='explore_subagent')
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
    assert sent.metadata['from'] == 'explore_subagent'
    assert collected.success is True
    assert collected.metadata['message_count'] == 1
    assert '<team_message>' in collected.content
    assert '<from>explore_subagent</from>' in collected.content
    assert 'Found the relevant files.' in collected.content
    assert empty.success is True
    assert empty.metadata['message_count'] == 0


def test_team_notification_escapes_xml_sensitive_content(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    message = bus.send(
        sender='lead',
        recipient='explore_subagent',
        message_type='warning',
        content='Use A < B & C > D',
    )

    rendered = render_team_notification((message,))[0]

    assert 'A &lt; B &amp; C &gt; D' in rendered
