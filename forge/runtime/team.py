'''Filesystem-backed team message bus for bounded subagent communication.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Literal
from uuid import uuid4


TeamMessageType = Literal[
    'status',
    'question',
    'result',
    'warning',
    'request',
    'response',
    'shutdown_request',
    'shutdown_response',
    'plan_approval_request',
    'plan_approval_response',
]
TeamRequestStatus = Literal['pending', 'approved', 'rejected']
TeamRequestType = Literal['shutdown', 'plan_approval', 'task_assignment', 'custom']


@dataclass(frozen=True, slots=True)
class TeamMessage:
    id: str
    sender: str
    recipient: str
    type: TeamMessageType
    content: str
    created_at: str
    request_id: str | None = None
    reply_to: str | None = None
    approve: bool | None = None

    def as_dict(self) -> dict[str, Any]:
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
            request_id=optional_str(data.get('request_id')),
            reply_to=optional_str(data.get('reply_to')),
            approve=optional_bool(data.get('approve')),
        )


@dataclass(frozen=True, slots=True)
class TeamRequest:
    request_id: str
    type: TeamRequestType
    sender: str
    target: str
    status: TeamRequestStatus
    payload: str
    created_at: str
    updated_at: str
    response: str = ''

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TeamRequest:
        return cls(
            request_id=str(data['request_id']),
            type=str(data['type']),  # type: ignore[arg-type]
            sender=str(data['sender']),
            target=str(data['target']),
            status=str(data['status']),  # type: ignore[arg-type]
            payload=str(data['payload']),
            created_at=str(data['created_at']),
            updated_at=str(data['updated_at']),
            response=str(data.get('response', '')),
        )


class MessageBus:
    '''Append-only team inbox plus request-state files under .forge/teams.'''

    def __init__(self, root: Path, *, team: str = 'default') -> None:
        self.root = root.resolve()
        self.team = clean_agent_id(team)
        self.base_directory = self.root / '.forge' / 'teams' / self.team
        self.directory = self.base_directory / 'inboxes'
        self.requests_directory = self.base_directory / 'requests'

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        message_type: TeamMessageType,
        content: str,
        request_id: str | None = None,
        reply_to: str | None = None,
        approve: bool | None = None,
    ) -> TeamMessage:
        clean_sender = clean_agent_id(sender)
        clean_recipient = clean_agent_id(recipient)
        clean_content = clean_message_content(content)
        clean_request_id = clean_request_id_value(request_id)
        clean_reply_to = clean_message_id(reply_to)
        message = TeamMessage(
            id=f'msg-{uuid4().hex[:12]}',
            sender=clean_sender,
            recipient=clean_recipient,
            type=message_type,
            content=clean_content,
            created_at=datetime.now(UTC).isoformat(),
            request_id=clean_request_id,
            reply_to=clean_reply_to,
            approve=approve,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(clean_recipient).open('a', encoding='utf-8') as file:
            file.write(json.dumps(message.as_dict(), ensure_ascii=False) + '\n')
        return message

    def request(
        self,
        *,
        sender: str,
        recipient: str,
        request_type: TeamRequestType,
        content: str,
    ) -> tuple[TeamRequest, TeamMessage]:
        clean_sender = clean_agent_id(sender)
        clean_recipient = clean_agent_id(recipient)
        clean_content = clean_message_content(content)
        request = TeamRequest(
            request_id=f'req-{uuid4().hex[:12]}',
            type=request_type,
            sender=clean_sender,
            target=clean_recipient,
            status='pending',
            payload=clean_content,
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._save_request(request)
        message = self.send(
            sender=clean_sender,
            recipient=clean_recipient,
            message_type=request_message_type(request_type),
            content=clean_content,
            request_id=request.request_id,
        )
        return request, message

    def respond(
        self,
        *,
        sender: str,
        request_id: str,
        approve: bool,
        content: str,
        reply_to: str | None = None,
    ) -> tuple[TeamRequest, TeamMessage]:
        clean_sender = clean_agent_id(sender)
        clean_content = clean_message_content(content)
        request = self.load_request(request_id)
        if request.status != 'pending':
            raise ValueError(f'Request {request_id} is already {request.status}.')
        if request.target != clean_sender:
            raise ValueError(
                f'Request {request_id} is targeted to {request.target}, '
                f'not {clean_sender}.'
            )
        status: TeamRequestStatus = 'approved' if approve else 'rejected'
        updated = replace(
            request,
            status=status,
            response=clean_content,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._save_request(updated)
        message = self.send(
            sender=clean_sender,
            recipient=request.sender,
            message_type=response_message_type(request.type),
            content=clean_content,
            request_id=updated.request_id,
            reply_to=reply_to,
            approve=approve,
        )
        return updated, message

    def list_requests(
        self,
        *,
        status: TeamRequestStatus | None = None,
    ) -> tuple[TeamRequest, ...]:
        if not self.requests_directory.exists():
            return ()
        requests: list[TeamRequest] = []
        for path in sorted(self.requests_directory.glob('req-*.json')):
            try:
                request = self._read_request(path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if status is None or request.status == status:
                requests.append(request)
        return tuple(requests)

    def load_request(self, request_id: str) -> TeamRequest:
        clean_request_id = clean_request_id_value(request_id)
        if clean_request_id is None:
            raise ValueError('Request ID must not be empty.')
        return self._read_request(self._request_path(clean_request_id))

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

    def _request_path(self, request_id: str) -> Path:
        clean_request_id = clean_request_id_value(request_id)
        if clean_request_id is None:
            raise ValueError('Request ID must not be empty.')
        return self.requests_directory / f'{clean_request_id}.json'

    def _save_request(self, request: TeamRequest) -> None:
        self.requests_directory.mkdir(parents=True, exist_ok=True)
        path = self._request_path(request.request_id)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(
            json.dumps(request.as_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        temporary.replace(path)

    @staticmethod
    def _read_request(path: Path) -> TeamRequest:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'Invalid team request file: {path}')
        return TeamRequest.from_dict(data)


def render_team_notification(messages: tuple[TeamMessage, ...]) -> tuple[str, ...]:
    return tuple(render_one_message(message) for message in messages)


def render_one_message(message: TeamMessage) -> str:
    request = (
        f'  <request_id>{message.request_id}</request_id>\n'
        if message.request_id
        else ''
    )
    reply = (
        f'  <reply_to>{message.reply_to}</reply_to>\n'
        if message.reply_to
        else ''
    )
    approve = (
        f'  <approve>{str(message.approve).lower()}</approve>\n'
        if message.approve is not None
        else ''
    )
    return (
        '<team_message>\n'
        f'  <id>{message.id}</id>\n'
        f'  <from>{message.sender}</from>\n'
        f'  <to>{message.recipient}</to>\n'
        f'  <type>{message.type}</type>\n'
        f'{request}'
        f'{reply}'
        f'{approve}'
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


def clean_request_id_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if re.fullmatch(r'req-[0-9a-f]{12}', cleaned) is None:
        raise ValueError(f'Invalid team request ID: {value}')
    return cleaned


def clean_message_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if re.fullmatch(r'msg-[0-9a-f]{12}', cleaned) is None:
        raise ValueError(f'Invalid team message ID: {value}')
    return cleaned


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if str(value).lower() == 'true':
        return True
    if str(value).lower() == 'false':
        return False
    return None


def request_message_type(request_type: TeamRequestType) -> TeamMessageType:
    if request_type == 'shutdown':
        return 'shutdown_request'
    if request_type == 'plan_approval':
        return 'plan_approval_request'
    return 'request'


def response_message_type(request_type: TeamRequestType) -> TeamMessageType:
    if request_type == 'shutdown':
        return 'shutdown_response'
    if request_type == 'plan_approval':
        return 'plan_approval_response'
    return 'response'


def escape_message_text(value: str) -> str:
    return (
        value.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
