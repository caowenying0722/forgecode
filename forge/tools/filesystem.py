'''Repository-scoped directory and file reading tools.'''

from __future__ import annotations

import asyncio

from pydantic import Field, model_validator

from forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)


class ListDirectoryInput(ToolInput):
    path: str = '.'


class ListDirectoryTool(Tool[ListDirectoryInput]):
    name = 'list_directory'
    description = 'List the direct children of a repository directory.'
    input_model = ListDirectoryInput

    async def execute(self, arguments: ListDirectoryInput) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, arguments)

    def _execute_sync(self, arguments: ListDirectoryInput) -> ToolResult:
        directory = resolve_repository_path(self.root, arguments.path)
        if not directory.is_dir():
            raise ToolExecutionError(
                'not_a_directory',
                f'Path is not a directory: {arguments.path}',
            )

        entries = sorted(
            directory.iterdir(),
            key=lambda path: (not path.is_dir(), path.name.casefold()),
        )
        lines = [
            f'{entry.name}/' if entry.is_dir() else entry.name
            for entry in entries
        ]
        shown_path = display_path(self.root, directory)
        return ToolResult.ok(
            f'Listed {len(entries)} entries in {shown_path}.',
            content='\n'.join(lines),
            metadata={'path': shown_path, 'entry_count': len(entries)},
        )


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def validate_line_range(self) -> ReadFileInput:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError('end_line must be greater than or equal to start_line')
        return self


class ReadFileTool(Tool[ReadFileInput]):
    name = 'read_file'
    description = (
        'Read a UTF-8 repository file with line numbers and an optional '
        'inclusive line range.'
    )
    input_model = ReadFileInput

    async def execute(self, arguments: ReadFileInput) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, arguments)

    def _execute_sync(self, arguments: ReadFileInput) -> ToolResult:
        path = resolve_repository_path(self.root, arguments.path)
        if not path.is_file():
            raise ToolExecutionError(
                'not_a_file',
                f'Path is not a file: {arguments.path}',
            )
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError as error:
            raise ToolExecutionError(
                'not_utf8_text',
                f'File is not valid UTF-8 text: {arguments.path}',
            ) from error

        total_lines = len(lines)
        start_index = min(arguments.start_line - 1, total_lines)
        end_line = (
            total_lines
            if arguments.end_line is None
            else min(arguments.end_line, total_lines)
        )
        selected = lines[start_index:end_line]
        numbered = [
            f'{line_number:>6} | {line}'
            for line_number, line in enumerate(
                selected,
                start=arguments.start_line,
            )
        ]
        shown_path = display_path(self.root, path)
        return ToolResult.ok(
            f'Read {len(selected)} lines from {shown_path}.',
            content='\n'.join(numbered),
            metadata={
                'path': shown_path,
                'start_line': arguments.start_line,
                'end_line': end_line,
                'total_lines': total_lines,
            },
        )
