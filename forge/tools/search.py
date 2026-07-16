'''Repository-scoped file discovery and text search tools.'''

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
import re

from pydantic import Field, field_validator

from forge.tools.base import (
    IGNORED_DIRECTORIES,
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(path):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in IGNORED_DIRECTORIES
            and not (Path(directory) / name).is_symlink()
        )
        for file_name in sorted(file_names):
            candidate = Path(directory) / file_name
            if not candidate.is_symlink():
                files.append(candidate)
    return files


class FindFilesInput(ToolInput):
    pattern: str = Field(min_length=1)
    path: str = '.'
    max_results: int = Field(default=200, ge=1, le=1000)


class FindFilesTool(Tool[FindFilesInput]):
    name = 'find_files'
    description = (
        'Find repository files by a glob pattern, excluding common generated '
        'directories.'
    )
    input_model = FindFilesInput

    async def execute(self, arguments: FindFilesInput) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, arguments)

    def _execute_sync(self, arguments: FindFilesInput) -> ToolResult:
        start = resolve_repository_path(self.root, arguments.path)
        matches: list[str] = []
        truncated = False
        for candidate in iter_files(start):
            relative = display_path(self.root, candidate)
            if (
                fnmatch.fnmatch(relative, arguments.pattern)
                or fnmatch.fnmatch(candidate.name, arguments.pattern)
            ):
                if len(matches) == arguments.max_results:
                    truncated = True
                    break
                matches.append(relative)

        return ToolResult.ok(
            f'Found {len(matches)} matching files.',
            content='\n'.join(matches),
            metadata={
                'pattern': arguments.pattern,
                'path': display_path(self.root, start),
                'match_count': len(matches),
                'truncated': truncated,
            },
        )


class GrepInput(ToolInput):
    pattern: str = Field(min_length=1)
    path: str = '.'
    file_types: list[str] = Field(default_factory=list)
    case_sensitive: bool = True
    regex: bool = True
    max_results: int = Field(default=200, ge=1, le=1000)

    @field_validator('file_types')
    @classmethod
    def normalize_file_types(cls, values: list[str]) -> list[str]:
        return [
            value.casefold() if value.startswith('.') else f'.{value.casefold()}'
            for value in values
        ]


class GrepTool(Tool[GrepInput]):
    name = 'grep'
    description = (
        'Search UTF-8 repository files and return path, line number, and '
        'matching text. Supports regex, path, and file type filters.'
    )
    input_model = GrepInput

    async def execute(self, arguments: GrepInput) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, arguments)

    def _execute_sync(self, arguments: GrepInput) -> ToolResult:
        start = resolve_repository_path(self.root, arguments.path)
        flags = 0 if arguments.case_sensitive else re.IGNORECASE
        expression = arguments.pattern if arguments.regex else re.escape(
            arguments.pattern
        )
        try:
            matcher = re.compile(expression, flags)
        except re.error as error:
            raise ToolExecutionError(
                'invalid_pattern',
                f'Invalid regular expression: {error}',
            ) from error

        matches: list[str] = []
        skipped_files = 0
        truncated = False
        for candidate in iter_files(start):
            if (
                arguments.file_types
                and candidate.suffix.casefold() not in arguments.file_types
            ):
                continue
            try:
                lines = candidate.read_text(encoding='utf-8').splitlines()
            except (UnicodeDecodeError, OSError):
                skipped_files += 1
                continue
            relative = display_path(self.root, candidate)
            for line_number, line in enumerate(lines, start=1):
                if matcher.search(line) is None:
                    continue
                if len(matches) == arguments.max_results:
                    truncated = True
                    break
                shown_line = line if len(line) <= 500 else f'{line[:497]}...'
                matches.append(f'{relative}:{line_number}:{shown_line}')
            if truncated:
                break

        return ToolResult.ok(
            f'Found {len(matches)} matching lines.',
            content='\n'.join(matches),
            metadata={
                'pattern': arguments.pattern,
                'path': display_path(self.root, start),
                'match_count': len(matches),
                'skipped_files': skipped_files,
                'truncated': truncated,
            },
        )
