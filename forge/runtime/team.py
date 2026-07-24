'''Filesystem-backed team message bus for bounded subagent communication.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Literal
from uuid import uuid4


TeamMessageType = Literal['status', 'question', 'result', 'warning']


@dataclass(frozen=True, slots=True)
class TeamMessage:
    id: str
    sender: str
    recipient: str
    type: TeamMessageType
    content: str
    created_at: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TeamMessage:
        return cls(
            id=str(data['id']),
            sender=str(data['sender']),
            recipient=str(data['recipient']),
            type=str(data['type']),  # type: ignore[arg-type]
            content=str(data['content']),
            created_at=str(data['created_at']),
        )


class MessageBus:
    '''Append-only inbox files under .forge/teams/<team>/inboxes.'''

    def __init__(self, root: Path, *, team: str = 'default') -> None:
        self.root = root.resolve()
        self.team = clean_agent_id(team)
        self.directory = self.root / '.forge' / 'teams' / self.team / 'inboxes'

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        message_type: TeamMessageType,
        content: str,
    ) -> TeamMessage:
        clean_sender = clean_agent_id(sender)
        clean_recipient = clean_agent_id(recipient)
        clean_content = clean_message_content(content)
        message = TeamMessage(
            id=f'msg-{uuid4().hex[:12]}',
            sender=clean_sender,
            recipient=clean_recipient,
            type=message_type,
            content=clean_content,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(clean_recipient).open('a', encoding='utf-8') as file:
            file.write(json.dumps(message.as_dict(), ensure_ascii=False) + '\n')
        return message

    def collect(self, recipient: str) -> tuple[TeamMessage, ...]:
        clean_recipient = clean_agent_id(recipient)
        path = self._path(clean_recipient)
        if not path.exists():
            return ()
        messages: list[TeamMessage] = []
        kept: list[str] = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    continue
                messages.append(TeamMessage.from_dict(data))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                kept.append(line)
        if kept:
            path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
        else:
            path.unlink()
        return tuple(messages)

    def _path(self, recipient: str) -> Path:
        return self.directory / f'{recipient}.jsonl'


def render_team_notification(messages: tuple[TeamMessage, ...]) -> tuple[str, ...]:
    return tuple(render_one_message(message) for message in messages)


def render_one_message(message: TeamMessage) -> str:
    return (
        '<team_message>\n'
        f'  <id>{message.id}</id>\n'
        f'  <from>{message.sender}</from>\n'
        f'  <to>{message.recipient}</to>\n'
        f'  <type>{message.type}</type>\n'
        f'  <content>{escape_message_text(message.content)}</content>\n'
        '</team_message>'
    )


def clean_agent_id(value: str) -> str:
    cleaned = str(value).strip()
    if re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', cleaned) is None:
        raise ValueError(
            'Agent IDs may contain only letters, numbers, dot, underscore, '
            'and dash, with length 1-80.'
        )
    return cleaned


def clean_message_content(value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError('Message content must not be empty.')
    if len(cleaned) > 8_000:
        raise ValueError('Message content is limited to 8000 characters.')
    return cleaned


def escape_message_text(value: str) -> str:
    return (
        value.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
