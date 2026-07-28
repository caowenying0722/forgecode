'''Model-visible message construction for the Agent runtime.'''

from __future__ import annotations

import json
from typing import Any

from forge.runtime.state import ToolCall
from forge.tools.base import ToolResult


def build_assistant_message(
    text: str,
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    if not tool_calls:
        return {'role': 'assistant', 'content': text}
    content: list[dict[str, Any]] = []
    if text:
        content.append({'type': 'text', 'text': text})
    content.extend(
        {
            'type': 'tool_use',
            'id': call.id,
            'name': call.name,
            'input': call.arguments,
        }
        for call in sorted(tool_calls, key=lambda call: call.index)
    )
    return {'role': 'assistant', 'content': content}


def build_tool_result_message(
    tool_results: list[tuple[ToolCall, ToolResult]],
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': [
            {
                'type': 'tool_result',
                'tool_use_id': tool_call.id,
                'content': serialize_tool_result(result),
                'is_error': not result.success,
            }
            for tool_call, result in tool_results
        ],
    }


def append_notification_message(
    messages: list[dict[str, Any]],
    notifications: tuple[str, ...],
) -> None:
    message = build_notification_message(notifications)
    if not messages or messages[-1].get('role') != 'user':
        messages.append(message)
        return
    existing = messages[-1].get('content')
    if isinstance(existing, str):
        messages[-1]['content'] = [
            {'type': 'text', 'text': existing},
            *message['content'],
        ]
    elif isinstance(existing, list):
        existing.extend(message['content'])
    else:
        messages.append(message)


def build_notification_message(
    notifications: tuple[str, ...],
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': notification}
            for notification in notifications
        ],
    }


def serialize_tool_result(result: ToolResult) -> str:
    error = None
    if result.error is not None:
        error = {
            'code': result.error.code,
            'message': result.error.message,
            'details': result.error.details,
        }
    return json.dumps(
        {
            'success': result.success,
            'summary': result.summary,
            'content': result.content,
            'error': error,
            'metadata': result.metadata,
        },
        ensure_ascii=False,
        default=str,
    )
