'''Model-visible recovery feedback and bounded failure evidence.'''

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from forge.runtime.state import ToolCall
from forge.runtime.tool_targets import mutation_target_paths
from forge.tools.base import ToolResult


def render_action_recovery_context(
    recovery_calls: int,
    maximum: int,
    *,
    read_used: bool,
) -> str:
    next_action = (
        'The one targeted repository read/search has already been used. '
        'Use the existing evidence and call a workspace editing tool now.'
        if read_used
        else (
            'If one exact code location is still missing, you may use one '
            'targeted read_file or grep call. Otherwise edit immediately.'
        )
    )
    return (
        '[ForgeCode Action Recovery]\n'
        'Investigation has consumed its bounded budget while the task-local '
        'Diff is still empty. This is a focused action phase. Use an editing '
        'tool now if the relevant code is understood. '
        f'{next_action} Broad diagnostics, process commands, Git inspection, '
        'verification, and task planning are intentionally unavailable until '
        'a real workspace revision is created. A preexisting Git Diff does '
        'not satisfy this turn. finish_task is valid only for a genuine '
        'external blocker.\n'
        f'Focused calls used: {recovery_calls}/{maximum}.'
    )


def build_action_recovery_feedback(
    task_context: str,
    recovery_calls: int,
    maximum: int,
    *,
    read_used: bool,
) -> dict[str, Any]:
    context = render_action_recovery_context(
        recovery_calls,
        maximum,
        read_used=read_used,
    )
    return {'role': 'user', 'content': f'{task_context}\n\n{context}'}


def action_recovery_stuck_reason(recovery_calls: int) -> str:
    return (
        f'Action Recovery stopped after {recovery_calls} focused model calls '
        'without a task-local workspace revision, although this turn '
        'requires a change.'
    )


def build_stagnation_feedback(
    calls_without_progress: int,
    task_context: str,
    working_context: str,
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n{working_context}\n\n'
            'ForgeCode progress check: '
            f'{calls_without_progress} model calls have passed since the '
            'last new workspace, task-plan, or repository evidence. All '
            'tools remain available. Reassess the root goal, use existing '
            'evidence, and choose a materially different next action. Do not '
            're-read paths already marked as fully covered. If the task needs '
            'a code change and the Diff is empty, edit the relevant code after '
            'you understand it; otherwise perform one targeted search for the '
            'specific missing fact. Do not repeat an unchanged failing action '
            'or claim tools are paused.'
        ),
    }


def build_stagnation_final_recovery_feedback(
    task_context: str,
    working_context: str,
    calls_without_progress: int,
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n{working_context}\n\n'
            '[ForgeCode Stagnation Final Recovery]\n'
            f'{calls_without_progress} model calls have passed without new '
            'workspace, plan, or repository evidence. The next request will '
            'include no tools. Return the best concise answer possible from '
            'the evidence already in context. If the goal is not actually '
            'complete, say what blocked completion and the exact next action '
            'that should be taken in a future tool-enabled turn. Do not '
            'request another tool call.'
        ),
    }


def build_token_limit_recovery_feedback(
    reason: str,
    task_context: str,
    working_context: str,
) -> dict[str, Any]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n{working_context}\n\n'
            '[ForgeCode Token-Limit Recovery]\n'
            f'{reason} The next request will include no tools. Return a '
            'concise user-facing progress summary from the evidence already '
            'in context. Include what was completed, what remains, any '
            'verification already performed, and the exact next action for a '
            'future tool-enabled turn. Do not request another tool call.'
        ),
    }


def mutation_failure_record(
    tool_call: ToolCall,
    result: ToolResult,
) -> dict[str, Any]:
    error_code = (
        result.error.code if result.error is not None else 'no_workspace_change'
    )
    message = (
        result.error.message
        if result.error is not None
        else (
            'The tool reported success, but the task-local workspace '
            'revision did not change.'
        )
    )
    diagnostic = result.content.strip()
    if len(diagnostic) > 2_000:
        diagnostic = (
            diagnostic[:1_000]
            + '\n...[diagnostic shortened]...\n'
            + diagnostic[-1_000:]
        )
    return {
        'tool': tool_call.name,
        'code': error_code,
        'message': message,
        'targets': list(mutation_target_paths(tool_call)[:5]),
        'diagnostic': diagnostic,
    }


def render_mutation_recovery_context(
    failures: list[dict[str, Any]],
    failure_count: int,
) -> str:
    lines = [
        '[Failed Mutation Recovery]',
        f'failed workspace writes: {failure_count}',
    ]
    for failure in failures:
        targets = ', '.join(failure['targets']) or 'unknown target'
        lines.append(
            f'- {failure["tool"]} [{failure["code"]}] on {targets}: '
            f'{failure["message"]}'
        )
        diagnostic = str(failure.get('diagnostic', '')).strip()
        if diagnostic:
            lines.append(f'  diagnostic: {diagnostic}')
    lines.append(mutation_recovery_instruction(failures))
    lines.append(
        'apply_patch accepts unified diff and Begin Patch; use replace_text '
        'for an exact change. Only a real workspace revision clears this.'
    )
    return '\n'.join(lines)


def mutation_recovery_instruction(
    failures: list[dict[str, Any]],
) -> str:
    latest = failures[-1] if failures else {}
    if str(latest.get('code', '')) in {
        'patch_rejected',
        'patch_apply_failed',
        'patch_context_not_found',
        'patch_context_ambiguous',
        'patch_contains_read_line_numbers',
    }:
        targets = ', '.join(str(item) for item in latest.get('targets', []))
        target_text = targets or 'the failed patch target'
        return (
            'The latest failure is PATCH_FAILED for '
            f'{target_text}. Do not restart broad discovery or switch to '
            'placeholder writes. Use at most one targeted read_file or grep '
            'for the failed target, then retry a smaller corrected patch or '
            'exact replacement against the same task-relevant code.'
        )
    if latest.get('code') == 'parent_not_found':
        parents = parent_directories_from_failures(failures)
        parent_text = ', '.join(parents) if parents else 'the missing parent'
        return (
            'The latest failure is parent_not_found. Do not retry another '
            'file write first. Call create_directory for '
            f'{parent_text}, then retry the original file write. Do not '
            'restart broad discovery, use placeholder writes, or patch '
            'unrelated files.'
        )
    return (
        'All normal tools remain available. Do not restart broad discovery. '
        'If the latest diagnostic includes Closest current text, copy it '
        'verbatim as the next old_text and do not re-read that region. Only '
        'when no exact candidate is supplied, make one targeted read before '
        'retrying a smaller corrected edit.'
    )


def parent_directories_from_failures(
    failures: list[dict[str, Any]],
) -> tuple[str, ...]:
    parents: list[str] = []
    for failure in failures:
        if failure.get('code') != 'parent_not_found':
            continue
        for target in failure.get('targets', []):
            if not isinstance(target, str) or not target.strip():
                continue
            parent = PurePosixPath(
                target.strip().replace('\\', '/')
            ).parent.as_posix()
            if parent and parent != '.':
                parents.append(parent)
    return tuple(dict.fromkeys(parents))[:5]


def build_mutation_recovery_feedback(
    failures: list[dict[str, Any]],
    failure_count: int,
    task_context: str,
) -> dict[str, Any]:
    context = render_mutation_recovery_context(failures, failure_count)
    return {'role': 'user', 'content': f'{task_context}\n\n{context}'}


def mutation_recovery_stuck_reason(
    failures: list[dict[str, Any]],
    failure_count: int,
) -> str:
    latest = failures[-1] if failures else {}
    tool = str(latest.get('tool', 'workspace tool'))
    code = str(latest.get('code', 'no_workspace_change'))
    return (
        f'Stopped after {failure_count} workspace-write attempt(s) failed '
        'to change the task workspace; the Edit Recovery failure limit was '
        f'reached. Latest failure: {tool} [{code}].'
    )
