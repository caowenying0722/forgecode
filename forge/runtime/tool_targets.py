'''Canonical extraction of workspace targets from model tool calls.'''

from __future__ import annotations

from forge.runtime.state import ToolCall
from forge.runtime.paths import normalize_workspace_path


def mutation_target_paths(tool_call: ToolCall) -> tuple[str, ...]:
    '''Return normalized paths referenced by one workspace mutation call.'''
    paths: list[str] = []
    arguments = tool_call.arguments
    for key in ('path', 'target_path'):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(normalize_workspace_path(value.strip()))
    for key in ('paths', 'changed_paths'):
        value = arguments.get(key)
        if isinstance(value, list):
            paths.extend(
                normalize_workspace_path(str(path).strip())
                for path in value
                if str(path).strip()
            )
    patch = arguments.get('patch')
    if isinstance(patch, str):
        paths.extend(paths_from_patch(patch))
    return tuple(dict.fromkeys(path for path in paths if path))


def paths_from_patch(patch: str) -> tuple[str, ...]:
    prefixes = (
        '*** Update File:',
        '*** Add File:',
        '*** Delete File:',
        '*** Move to:',
        '+++ b/',
        '--- a/',
    )
    paths: list[str] = []
    for line in patch.splitlines():
        stripped = line.strip()
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if stripped.startswith(candidate)
            ),
            None,
        )
        if prefix is None:
            continue
        path = normalize_workspace_path(stripped[len(prefix):].strip())
        if path and path != '/dev/null':
            paths.append(path)
    return tuple(dict.fromkeys(paths))
