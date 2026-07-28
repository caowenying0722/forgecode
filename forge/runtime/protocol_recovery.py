'''Feedback for malformed model/tool protocol responses.'''

from __future__ import annotations

from typing import Any

from forge.runtime.model_client import ModelProtocolError
from forge.runtime.state import ToolCall
from forge.tools.base import ToolResult


def build_tool_protocol_feedback(
    failures: int,
    task_context: str,
    tool_results: list[tuple[ToolCall, ToolResult]] | None = None,
) -> dict[str, Any]:
    diagnostics: list[str] = []
    for tool_call, result in tool_results or ():
        if result.error is None:
            continue
        message = result.error.message
        if len(message) > 1_500:
            message = f'{message[:1_497]}...'
        diagnostics.append(f'- {tool_call.name}: {message}')
    rendered = (
        '\nExact rejection(s):\n' + '\n'.join(diagnostics) + '\n'
        if diagnostics
        else ''
    )
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n'
            'The previous tool request was rejected at the argument/schema '
            'boundary. This does not mean the repository task is blocked. '
            f'{rendered}'
            'Follow the exact recovery instruction above, change the '
            'arguments materially, and retry with valid JSON or choose '
            'another tool. Do not repeat the rejected payload. '
            f'Protocol recovery count: {failures}.'
        ),
    }


def build_synthesis_retry_feedback(
    task_context: str,
    working_context: str,
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n{working_context}\n\n'
            'ForgeCode rejected the previous synthesis because it did not '
            'reference collected repository evidence. All tools remain '
            'available. Answer the current goal using the working evidence, '
            'or gather genuinely missing evidence before answering.'
        ),
    }


def build_output_continuation_feedback(
    *,
    attempt: int,
    maximum: int,
) -> dict[str, str]:
    return {
        'role': 'user',
        'content': (
            'The previous response reached the output token limit. The text '
            'already generated has been preserved. Continue directly from '
            'where it stopped without repeating earlier content, and finish '
            'concisely. If work remains, use the available tools instead of '
            'printing large code blocks. '
            f'Continuation attempt {attempt} of {maximum}.'
        ),
    }


def build_protocol_recovery_feedback(
    error: ModelProtocolError,
    *,
    attempt: int,
    maximum: int,
    available_tools: tuple[str, ...],
) -> list[dict[str, Any]]:
    tool = f' for tool {error.tool_name!r}' if error.tool_name else ''
    if error.reason == 'output_truncated':
        problem = 'The previous response reached the max_tokens limit.'
    elif error.reason == 'unavailable_tool':
        problem = f'The previous response requested unavailable tool{tool}.'
    else:
        problem = f'The previous tool call{tool} had invalid arguments.'
    available = ', '.join(available_tools) if available_tools else 'none'
    retry_limit = 4_000 if attempt == 1 else 2_000
    retry_strategy = (
        'Modify only one function or one file section.'
        if attempt == 1
        else (
            'Create only a minimal skeleton. Keep HTML, CSS, and JavaScript '
            'in separate tool calls.'
        )
    )
    return [
        {
            'role': 'assistant',
            'content': '[ForgeCode rejected an invalid model response.]',
        },
        {
            'role': 'user',
            'content': (
                f'{problem}\nError: {error}\n'
                'No tool was executed and no file was changed by that '
                f'response. Available tools: {available}. For a small complete '
                f'file, use write_file with at most {retry_limit} characters. '
                'For a focused exact change, use replace_text. For structured '
                f'edits, use apply_patch with at most {retry_limit} characters. '
                f'{retry_strategy} Split large HTML, CSS, or JavaScript across '
                'multiple calls and do not repeat the same invalid arguments.\n'
                f'Recovery attempt {attempt} of {maximum}.'
            ),
        },
    ]
