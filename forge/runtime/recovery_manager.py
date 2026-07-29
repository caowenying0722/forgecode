'''Recovery tool-selection policies for the Agent Loop.'''

from __future__ import annotations

from typing import Any

from forge.runtime.tool_runner import ToolRunner


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
        if verify_available:
            allowed.add('verify')
        if fix_available:
            allowed.add('run_command')
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
