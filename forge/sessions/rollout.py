'''Crash-tolerant append-only session rollout storage.'''

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Iterable


class SessionRollout:
    '''Persist replayable session events next to materialized snapshots.'''

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._next_sequences: dict[str, int] = {}

    def path_for(self, session_id: str) -> Path:
        return self.directory / f'{session_id}.rollout.jsonl'

    def append(
        self,
        session_id: str,
        records: Iterable[tuple[str, dict[str, Any]]],
    ) -> int:
        pending = list(records)
        if not pending:
            return self.last_sequence(session_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        sequence = self._next_sequence(session_id)
        path = self.path_for(session_id)
        with path.open('a', encoding='utf-8') as file:
            for event_type, payload in pending:
                record = {
                    'seq': sequence,
                    'timestamp': datetime.now().astimezone().isoformat(),
                    'type': event_type,
                    'payload': to_json_value(payload),
                }
                file.write(
                    json.dumps(record, ensure_ascii=False, default=str)
                )
                file.write('\n')
                sequence += 1
            file.flush()
            os.fsync(file.fileno())
        self._next_sequences[session_id] = sequence
        return sequence - 1

    def read(self, session_id: str) -> tuple[dict[str, Any], ...]:
        path = self.path_for(session_id)
        if not path.is_file():
            return ()
        lines = path.read_text(encoding='utf-8').splitlines()
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise ValueError(
                    f'Corrupt session rollout at line {index + 1}: {path}'
                ) from None
            if isinstance(value, dict):
                records.append(value)
        return tuple(records)

    def last_sequence(self, session_id: str) -> int:
        records = self.read(session_id)
        if not records:
            return 0
        try:
            return int(records[-1].get('seq', 0))
        except (TypeError, ValueError):
            return 0

    def replay(
        self,
        session_id: str,
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = self.read(session_id)
        state = dict(fallback or {})
        messages = (
            []
            if records
            else json_round_trip(state.get('messages', []))
        )
        completed_tools: dict[str, dict[str, Any]] = {}
        started_tools: dict[str, dict[str, Any]] = {}
        for record in records:
            event_type = str(record.get('type', ''))
            payload = record.get('payload')
            if not isinstance(payload, dict):
                continue
            if event_type == 'session_started':
                state.update(payload)
            elif event_type == 'messages_appended':
                additions = payload.get('messages', [])
                if isinstance(additions, list):
                    messages.extend(
                        decode_message_additions(
                            additions,
                            completed_tools,
                        )
                    )
            elif event_type == 'messages_replaced':
                replacement = payload.get('messages', [])
                if isinstance(replacement, list):
                    messages = json_round_trip(replacement)
            elif event_type == 'session_state':
                state.update(payload)
            elif event_type == 'tool_execution_started':
                tool_id = str(payload.get('tool_call_id', ''))
                if tool_id:
                    started_tools[tool_id] = payload
            elif event_type == 'tool_execution_completed':
                tool_id = str(payload.get('tool_call_id', ''))
                if tool_id:
                    completed_tools[tool_id] = payload
        state['messages'] = repair_dangling_tool_batch(
            messages,
            started_tools,
            completed_tools,
        )
        return state

    def _next_sequence(self, session_id: str) -> int:
        cached = self._next_sequences.get(session_id)
        if cached is not None:
            return cached
        sequence = self.last_sequence(session_id) + 1
        self._next_sequences[session_id] = sequence
        return sequence


def runtime_event_record(event: object) -> tuple[str, dict[str, Any]]:
    name = type(event).__name__
    event_type = camel_to_snake(name)
    payload = asdict(event) if is_dataclass(event) else {'value': str(event)}
    normalized = to_json_value(payload)
    if event_type in {'tool_execution_started', 'tool_execution_completed'}:
        tool_call = normalized.get('tool_call', {})
        normalized['tool_call_id'] = str(tool_call.get('id', ''))
        normalized['name'] = str(tool_call.get('name', ''))
    return event_type, normalized


def message_delta_records(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    reference_tool_results: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    if len(current) >= len(previous) and current[:len(previous)] == previous:
        additions = current[len(previous):]
        if not additions:
            return []
        return [
            (
                'messages_appended',
                {
                    'messages': (
                        encode_message_additions(additions)
                        if reference_tool_results
                        else additions
                    )
                },
            )
        ]
    if current == previous:
        return []
    return [('messages_replaced', {'messages': current})]


def repair_dangling_tool_batch(
    messages: list[dict[str, Any]],
    started_tools: dict[str, dict[str, Any]],
    completed_tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    '''Close a persisted assistant tool-use batch after an interrupted turn.'''
    if not messages:
        return messages
    assistant = messages[-1]
    if assistant.get('role') != 'assistant':
        return messages
    content = assistant.get('content')
    if not isinstance(content, list):
        return messages
    tool_uses = [
        item
        for item in content
        if isinstance(item, dict) and item.get('type') == 'tool_use'
    ]
    if not tool_uses:
        return messages
    results: list[dict[str, Any]] = []
    for tool_use in tool_uses:
        tool_id = str(tool_use.get('id', ''))
        completed = completed_tools.get(tool_id)
        if completed is not None:
            result = completed.get('result')
            if not isinstance(result, dict):
                result = interrupted_result(tool_id, started=False)
        else:
            result = interrupted_result(
                tool_id,
                started=tool_id in started_tools,
            )
        results.append(
            {
                'type': 'tool_result',
                'tool_use_id': tool_id,
                'content': json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                ),
                'is_error': not bool(result.get('success')),
            }
        )
    return [
        *messages,
        {'role': 'user', 'content': results},
    ]


def encode_message_additions(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for message in messages:
        content = message.get('content')
        if (
            message.get('role') == 'user'
            and isinstance(content, list)
            and content
            and all(
                isinstance(item, dict)
                and item.get('type') == 'tool_result'
                for item in content
            )
        ):
            encoded.append(
                {
                    '$tool_results': [
                        str(item.get('tool_use_id', ''))
                        for item in content
                    ]
                }
            )
        else:
            encoded.append(message)
    return encoded


def decode_message_additions(
    messages: list[Any],
    completed_tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        references = message.get('$tool_results')
        if not isinstance(references, list):
            decoded.append(json_round_trip(message))
            continue
        content: list[dict[str, Any]] = []
        for raw_id in references:
            tool_id = str(raw_id)
            completed = completed_tools.get(tool_id, {})
            result = completed.get('result')
            if not isinstance(result, dict):
                result = interrupted_result(tool_id, started=True)
            content.append(tool_result_block(tool_id, result))
        decoded.append({'role': 'user', 'content': content})
    return decoded


def tool_result_block(tool_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        'type': 'tool_result',
        'tool_use_id': tool_id,
        'content': json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        ),
        'is_error': not bool(result.get('success')),
    }


def interrupted_result(tool_id: str, *, started: bool) -> dict[str, Any]:
    state = 'started but did not record a result' if started else 'did not start'
    message = (
        f'Tool call {tool_id} {state} before the prior process stopped. '
        'Inspect the current workspace before deciding whether to retry it.'
    )
    return {
        'success': False,
        'summary': 'The previous ForgeCode process stopped mid-tool batch.',
        'content': '',
        'error': {
            'code': 'interrupted_tool_call',
            'message': message,
            'details': {'started': started},
        },
        'metadata': {'interrupted': True},
    }


def camel_to_snake(value: str) -> str:
    characters: list[str] = []
    for index, character in enumerate(value):
        if character.isupper() and index:
            characters.append('_')
        characters.append(character.casefold())
    return ''.join(characters)


def to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
