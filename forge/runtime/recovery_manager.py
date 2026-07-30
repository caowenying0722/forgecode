'''Recovery tool-selection policies for the Agent Loop.'''

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any

from forge.runtime.state import ToolCall, VerificationEvidence
from forge.runtime.tool_targets import mutation_target_paths
from forge.runtime.tool_runner import ToolRunner
from forge.tools.base import ToolResult


MAX_DIAGNOSTIC_EXCERPT = 1_000

PATH_LINE_PATTERNS = (
    re.compile(
        r'(?P<path>[A-Za-z0-9_.\\/-]+\.[A-Za-z0-9_]+)'
        r'[:(](?P<line>\d+)(?::\d+)?'
    ),
    re.compile(
        r'File "(?P<path>[^"]+)", line (?P<line>\d+)',
    ),
)
SYMBOL_PATTERNS = (
    re.compile(r"name '([^']+)' is not defined"),
    re.compile(r"Cannot find name '([^']+)'"),
    re.compile(r'undefined: ([A-Za-z_][A-Za-z0-9_]*)'),
    re.compile(r'NameError:\s+(?!name\b)([A-Za-z_][A-Za-z0-9_]*)'),
    re.compile(r"TS2305: Module '[^']+' has no exported member '([^']+)'"),
)
MODULE_PATTERNS = (
    re.compile(r"TS2305: Module '([^']+)' has no exported member"),
    re.compile(r"Cannot find module '([^']+)'"),
)


@dataclass(frozen=True, slots=True)
class RepairTarget:
    '''Specific repair focus derived from the latest structured failure.'''

    source: str
    expected_action: str
    paths: tuple[str, ...] = ()
    line_numbers: tuple[int, ...] = ()
    symbols: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    missing_exports: tuple[str, ...] = ()
    direct_dependencies: tuple[str, ...] = ()
    failure_signature: str = ''
    diagnostic_excerpt: str = ''
    baseline_source_revision: int | None = None
    read_ranges: tuple[str, ...] = ()
    attempted_edits: tuple[str, ...] = ()

    @property
    def has_specific_location(self) -> bool:
        return bool(self.paths or self.line_numbers or self.symbols)


class RecoveryManager:
    '''Select the smallest safe tool surface for a recovery request.'''

    def __init__(
        self,
        tools: list[dict[str, Any]] | None,
        tool_runner: ToolRunner | None,
        *,
        read_tools: frozenset[str],
        excluded_write_tools: frozenset[str],
    ) -> None:
        self.tools = tools
        self.tool_runner = tool_runner
        self.read_tools = read_tools
        self.excluded_write_tools = excluded_write_tools

    def action_tools(
        self,
        *,
        read_available: bool,
        include_finish: bool = True,
    ) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        selected: list[dict[str, Any]] = []
        for definition in self.tools:
            name = str(definition.get('name', ''))
            if (
                (read_available and name in self.read_tools)
                or (include_finish and name == 'finish_task')
                or (
                    self.tool_runner is not None
                    and self.tool_runner.effect(name) == 'workspace_write'
                    and name not in self.excluded_write_tools
                )
            ):
                selected.append(definition)
        return selected

    def mutation_tools(
        self,
        failures: list[dict[str, Any]],
        *,
        read_available: bool,
        include_finish: bool = False,
    ) -> list[dict[str, Any]] | None:
        latest = failures[-1] if failures else {}
        latest_code = str(latest.get('code', ''))
        if latest_code in {
            'patch_rejected',
            'patch_apply_failed',
            'patch_context_not_found',
            'patch_context_ambiguous',
            'patch_contains_read_line_numbers',
        }:
            return self.scoped_mutation_tools(
                failures,
                read_available=read_available,
                include_finish=include_finish,
            )
        if latest.get('code') != 'parent_not_found':
            return self.action_tools(
                read_available=read_available,
                include_finish=include_finish,
            )
        if self.tools is None:
            return None
        failed_write_tools = {
            str(failure.get('tool', ''))
            for failure in failures
            if str(failure.get('code', '')) == 'parent_not_found'
        }
        allowed = {
            'create_directory',
            *(failed_write_tools & {
                'apply_patch', 'write_file', 'write_file_chunk'
            }),
        }
        if not allowed & {'apply_patch', 'write_file', 'write_file_chunk'}:
            allowed.update({'write_file', 'write_file_chunk', 'apply_patch'})
        if read_available:
            allowed.add('list_directory')
        return [
            definition
            for definition in self.tools
            if str(definition.get('name', '')) in allowed
        ]

    def scoped_mutation_tools(
        self,
        failures: list[dict[str, Any]],
        *,
        read_available: bool,
        include_finish: bool,
    ) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        failed_tools = {
            str(failure.get('tool', ''))
            for failure in failures
            if str(failure.get('tool', ''))
        }
        allowed = {
            tool
            for tool in failed_tools
            if (
                tool in {
                    'apply_patch',
                    'replace_text',
                    'write_file',
                    'write_file_chunk',
                }
                or (
                    self.tool_runner is not None
                    and self.tool_runner.effect(tool) == 'workspace_write'
                    and tool not in self.excluded_write_tools
                )
            )
        }
        if not allowed:
            allowed.add('apply_patch')
        if read_available:
            allowed.update({'read_file', 'grep'})
        if include_finish:
            allowed.add('finish_task')
        return [
            definition
            for definition in self.tools
            if str(definition.get('name', '')) in allowed
        ]

    def verification_tools(
        self,
        *,
        fix_available: bool,
        read_available: bool,
        verify_available: bool = True,
    ) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        allowed: set[str] = set()
        allowed.add('finish_task')
        if verify_available:
            allowed.add('verify')
            allowed.update({'git_status', 'git_diff'})
        if fix_available:
            if read_available:
                allowed.update({'find_files', 'grep', 'read_file'})
        return [
            definition
            for definition in self.tools
            if (
                str(definition.get('name', '')) in allowed
                or (
                    fix_available
                    and
                    self.tool_runner is not None
                    and self.tool_runner.effect(str(definition.get('name', '')))
                    == 'workspace_write'
                    and str(definition.get('name', ''))
                    not in self.excluded_write_tools
                )
            )
        ]

    def finalization_tools(self) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        return [
            definition
            for definition in self.tools
            if str(definition.get('name', '')) == 'finish_task'
        ]

    def planning_tools(self) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        return [
            definition
            for definition in self.tools
            if str(definition.get('name', '')) == 'todo_write'
        ]

    def mutation_repair_target(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> RepairTarget:
        return repair_target_from_tool_failure(tool_call, result)

    def verification_repair_target(
        self,
        verification: VerificationEvidence,
        *,
        changed_paths: tuple[str, ...],
    ) -> RepairTarget:
        return repair_target_from_verification(
            verification,
            changed_paths=changed_paths,
        )

    def verification_repair_target_from_result(
        self,
        result: ToolResult,
        *,
        changed_paths: tuple[str, ...],
    ) -> RepairTarget:
        return repair_target_from_verification_result(
            result,
            changed_paths=changed_paths,
        )

    def verification_read_budget(self, target: RepairTarget | None) -> int:
        if target is None:
            return 2
        focus_count = len(
            set(
                [
                    *target.paths,
                    *target.symbols,
                    *target.modules,
                    *target.missing_exports,
                    *target.direct_dependencies,
                ]
            )
        )
        range_count = len(target.line_numbers)
        return max(2, min(8, focus_count + range_count or 2))


def repair_target_from_tool_failure(
    tool_call: ToolCall,
    result: ToolResult,
) -> RepairTarget:
    code = result.error.code if result.error is not None else 'no_workspace_change'
    diagnostic = _diagnostic_text(result)
    paths = tuple(
        dict.fromkeys(
            [
                *mutation_target_paths(tool_call)[:5],
                *_extract_paths(diagnostic),
            ]
        )
    )
    return RepairTarget(
        source=f'{tool_call.name}:{code}',
        expected_action=_expected_action_for_tool_failure(code),
        paths=paths,
        line_numbers=_extract_line_numbers(diagnostic),
        symbols=_extract_symbols(diagnostic),
        modules=_extract_modules(diagnostic),
        missing_exports=_extract_missing_exports(diagnostic),
        failure_signature=str(result.metadata.get('failure_signature', '')),
        diagnostic_excerpt=_excerpt(diagnostic),
        baseline_source_revision=_metadata_source_revision(result),
    )


def repair_target_from_verification(
    verification: VerificationEvidence,
    *,
    changed_paths: tuple[str, ...],
) -> RepairTarget:
    diagnostic = '\n'.join(
        part
        for part in (
            verification.command,
            verification.failure_signature,
        )
        if part
    )
    paths = _changed_paths_for_verification(changed_paths)
    modules = _extract_modules(diagnostic)
    return RepairTarget(
        source=f'verify:{verification.status}',
        expected_action=_expected_action_for_verification(verification),
        paths=paths,
        modules=modules,
        direct_dependencies=_direct_dependencies(paths, modules),
        failure_signature=verification.failure_signature,
        diagnostic_excerpt=_excerpt(diagnostic),
        baseline_source_revision=verification.bound_source_revision,
    )


def repair_target_from_verification_result(
    result: ToolResult,
    *,
    changed_paths: tuple[str, ...],
) -> RepairTarget:
    status = str(result.metadata.get('verification_status', 'failed'))
    diagnostic = _diagnostic_text(result)
    paths = tuple(
        dict.fromkeys(
            [
                *_extract_paths(diagnostic),
                *_changed_paths_for_verification(changed_paths),
            ]
        )
    )
    modules = _extract_modules(diagnostic)
    return RepairTarget(
        source=f'verify:{status}',
        expected_action=_expected_action_for_verification_status(status),
        paths=paths,
        line_numbers=_extract_line_numbers(diagnostic),
        symbols=_extract_symbols(diagnostic),
        modules=modules,
        missing_exports=_extract_missing_exports(diagnostic),
        direct_dependencies=_direct_dependencies(paths, modules),
        failure_signature=_failure_signature_from_result(result),
        diagnostic_excerpt=_excerpt(diagnostic),
        baseline_source_revision=_metadata_source_revision(result),
    )


def render_repair_target_context(target: RepairTarget | None) -> str:
    if target is None:
        return ''
    lines = [
        '[ForgeCode Repair Target]',
        f'- source: {target.source}',
        f'- expected action: {target.expected_action}',
    ]
    if target.paths:
        lines.append(f'- files: {", ".join(target.paths)}')
    if target.line_numbers:
        lines.append(
            '- lines: '
            + ', '.join(str(line) for line in target.line_numbers)
        )
    if target.symbols:
        lines.append(f'- symbols: {", ".join(target.symbols)}')
    if target.missing_exports:
        lines.append(
            f'- missing exports: {", ".join(target.missing_exports)}'
        )
    if target.modules:
        lines.append(f'- modules: {", ".join(target.modules)}')
    if target.direct_dependencies:
        lines.append(
            '- direct dependencies: '
            + ', '.join(target.direct_dependencies)
        )
    if target.failure_signature:
        lines.append(f'- failure signature: {target.failure_signature}')
    if target.diagnostic_excerpt:
        lines.append(f'- diagnostic: {target.diagnostic_excerpt}')
    if target.baseline_source_revision is not None:
        lines.append(
            f'- baseline source revision: {target.baseline_source_revision}'
        )
    lines.append(
        'Use this target before any broad discovery. You may read multiple '
        'small ranges inside these files and direct dependencies. Do not edit '
        'until the latest target content for the current source revision has '
        'been read; otherwise read the minimal current hunk first.'
    )
    return '\n'.join(lines)


def _diagnostic_text(result: ToolResult) -> str:
    parts = [
        result.summary,
        result.content,
    ]
    if result.error is not None:
        parts.append(result.error.message)
        parts.extend(str(value) for value in result.error.details.values())
    return '\n'.join(part for part in parts if part)


def _metadata_source_revision(result: ToolResult) -> int | None:
    try:
        return int(
            result.metadata.get(
                'source_revision',
                result.metadata.get('workspace_revision'),
            )
        )
    except (TypeError, ValueError):
        return None


def _extract_paths(text: str) -> tuple[str, ...]:
    paths: list[str] = []
    for pattern in PATH_LINE_PATTERNS:
        for match in pattern.finditer(text):
            path = match.group('path').replace('\\', '/')
            if path not in paths:
                paths.append(path)
    return tuple(paths[:8])


def _extract_line_numbers(text: str) -> tuple[int, ...]:
    lines: list[int] = []
    for pattern in PATH_LINE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                line = int(match.group('line'))
            except ValueError:
                continue
            if line not in lines:
                lines.append(line)
    return tuple(lines[:8])


def _extract_symbols(text: str) -> tuple[str, ...]:
    symbols: list[str] = []
    for pattern in SYMBOL_PATTERNS:
        for match in pattern.finditer(text):
            symbol = match.group(1)
            if symbol not in symbols:
                symbols.append(symbol)
    return tuple(symbols[:8])


def _extract_modules(text: str) -> tuple[str, ...]:
    modules: list[str] = []
    for pattern in MODULE_PATTERNS:
        for match in pattern.finditer(text):
            module = match.group(1)
            if module not in modules:
                modules.append(module)
    return tuple(modules[:8])


def _extract_missing_exports(text: str) -> tuple[str, ...]:
    exports: list[str] = []
    pattern = re.compile(
        r"TS2305: Module '[^']+' has no exported member '([^']+)'"
    )
    for match in pattern.finditer(text):
        export = match.group(1)
        if export not in exports:
            exports.append(export)
    return tuple(exports[:8])


def _direct_dependencies(
    paths: tuple[str, ...],
    modules: tuple[str, ...],
) -> tuple[str, ...]:
    dependencies: list[str] = []
    for path in paths:
        if path not in dependencies:
            dependencies.append(path)
    for module in modules:
        if module.startswith('.'):
            parents = tuple(
                PurePosixPath(path).parent.as_posix()
                for path in paths
                if PurePosixPath(path).parent.as_posix() != '.'
            )
            bases = parents or ('',)
            for suffix in ('.ts', '.tsx', '.js', '.jsx'):
                for base in bases:
                    candidate = PurePosixPath(base, module).as_posix()
                    candidate = candidate.removeprefix('./') + suffix
                    if candidate not in dependencies:
                        dependencies.append(candidate)
        elif module not in dependencies:
            dependencies.append(module)
    return tuple(dependencies[:8])


def _changed_paths_for_verification(
    changed_paths: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(path.replace('\\', '/') for path in changed_paths[:8])


def _expected_action_for_tool_failure(code: str) -> str:
    if code == 'parent_not_found':
        return 'create the missing parent directory, then retry the write'
    if code in {
        'patch_rejected',
        'patch_apply_failed',
        'patch_context_not_found',
        'patch_context_ambiguous',
        'patch_contains_read_line_numbers',
    }:
        return 'retry a smaller corrected patch against the same target'
    if code == 'invalid_arguments':
        return 'correct the tool arguments before retrying'
    return 'repair the failed workspace edit target'


def _expected_action_for_verification(
    verification: VerificationEvidence,
) -> str:
    return _expected_action_for_verification_status(verification.status)


def _expected_action_for_verification_status(status: str) -> str:
    if status == 'invalid':
        return 'choose a valid non-interactive validation command'
    if status == 'timed_out':
        return 'narrow or replace the timed-out verification command'
    if status == 'unavailable':
        return 'add or discover a project validation command'
    return 'repair the failing changed code or project configuration'


def _excerpt(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= MAX_DIAGNOSTIC_EXCERPT:
        return stripped
    head = MAX_DIAGNOSTIC_EXCERPT // 2
    tail = MAX_DIAGNOSTIC_EXCERPT - head
    return (
        stripped[:head]
        + '\n...[diagnostic shortened]...\n'
        + stripped[-tail:]
    )


def _failure_signature_from_result(result: ToolResult) -> str:
    explicit = str(result.metadata.get('failure_signature', ''))
    if explicit:
        return explicit
    if result.success:
        return ''
    signature_text = '\n'.join(
        str(result.metadata.get(key, ''))
        for key in ('command', 'exit_code', 'stderr')
    )
    if not signature_text.strip():
        signature_text = _diagnostic_text(result)
    return hashlib.sha256(signature_text.encode('utf-8')).hexdigest()
