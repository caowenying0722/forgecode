'''Pure helpers for Agent Loop tool-call identity and preflight checks.'''

from __future__ import annotations

import hashlib
import json
from typing import Any

from forge.runtime.state import ToolCall
from forge.runtime.task_scope import TaskScope, evaluate_change_relevance
from forge.runtime.tool_targets import mutation_target_paths
from forge.tools.base import ToolResult


def tool_call_signature(tool_call: ToolCall, revision: int) -> str:
    '''Identify an exact tool request within one workspace revision.'''
    arguments = json.dumps(
        normalize_tool_arguments(tool_call.name, tool_call.arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    digest = hashlib.sha256(arguments.encode('utf-8')).hexdigest()[:24]
    return f'{revision}:{tool_call.name}:{digest}'


def early_mutation_relevance_failure(
    tool_call: ToolCall,
    *,
    tool_effect: str | None,
    change_required: bool,
    task_scope_patterns: tuple[str, ...],
    task_scope_sources: tuple[str, ...] = (),
) -> ToolResult | None:
    '''Block statically obvious off-goal workspace edits before execution.'''
    if (
        not change_required
        or tool_effect != 'workspace_write'
        or not task_scope_patterns
    ):
        return None
    targets = mutation_target_paths(tool_call)
    if not targets:
        return None
    relevance_targets = _scope_probe_paths(targets)
    relevance = evaluate_change_relevance(
        relevance_targets,
        TaskScope(patterns=task_scope_patterns),
    )
    if relevance.relevant:
        return None
    return ToolResult.fail(
        'irrelevant_mutation_target',
        (
            f'{tool_call.name} targets paths outside the current task scope: '
            + ', '.join(targets)
            + '. Choose a task-relevant edit target instead.'
        ),
        details={
            'targets': list(targets),
            'task_scope_patterns': list(task_scope_patterns[:16]),
            'scope_sources': list(task_scope_sources),
            'reasons': list(relevance.reasons),
        },
        metadata={'irrelevant_mutation_target': True},
    )


def normalize_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    defaults: dict[str, Any] = {}
    if tool_name == 'read_file':
        defaults = {'start_line': 1, 'end_line': None}
    elif tool_name == 'list_directory':
        defaults = {'path': '.', 'max_results': 1000}
    elif tool_name == 'find_files':
        defaults = {'path': '.', 'max_results': 200}
    elif tool_name == 'grep':
        defaults = {
            'path': '.',
            'file_types': [],
            'case_sensitive': True,
            'regex': True,
            'max_results': 200,
        }
    elif tool_name == 'verify':
        defaults = {
            'target': 'auto',
            'command_id': '',
            'command': '',
            'cwd': '.',
            'timeout_seconds': 120.0,
        }
    elif tool_name == 'git_diff':
        defaults = {'path': None, 'cached': False}

    for key, value in defaults.items():
        normalized.setdefault(key, value)
    for key in ('path', 'cwd'):
        if key in normalized and normalized[key] is not None:
            normalized[key] = normalize_signature_path(str(normalized[key]))
    if tool_name == 'grep' and isinstance(normalized.get('file_types'), list):
        normalized['file_types'] = sorted(
            normalize_file_type(str(item))
            for item in normalized['file_types']
        )
    return normalized


def normalize_signature_path(path: str) -> str:
    rendered = path.strip().replace('\\', '/')
    while rendered.startswith('./'):
        rendered = rendered[2:]
    rendered = rendered.rstrip('/')
    return rendered or '.'


def normalize_file_type(value: str) -> str:
    lowered = value.strip().casefold()
    if not lowered:
        return lowered
    return lowered if lowered.startswith('.') else f'.{lowered}'


def _scope_probe_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for path in paths:
        normalized = path.replace('\\', '/').rstrip('/')
        expanded.append(normalized)
        name = normalized.rsplit('/', 1)[-1]
        if '.' not in name:
            expanded.append(f'{normalized}/__forge_scope_probe__')
    return tuple(dict.fromkeys(expanded))
