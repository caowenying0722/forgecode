'''Read-only Git status and diff tools.'''

from __future__ import annotations

from pydantic import Field

from forge.tools.base import (
    Tool,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)
from forge.tools.shell import (
    process_metadata,
    render_process_output,
    run_process,
)


class GitStatusInput(ToolInput):
    pass


class GitStatusTool(Tool[GitStatusInput]):
    name = 'git_status'
    description = 'Show the repository branch and concise working tree status.'
    input_model = GitStatusInput

    async def execute(self, arguments: GitStatusInput) -> ToolResult:
        result = await run_process(
            ['git', 'status', '--short', '--branch'],
            cwd=self.root,
            timeout_seconds=30,
        )
        metadata = process_metadata(result)
        if result.exit_code != 0:
            return ToolResult.fail(
                'git_status_failed',
                f'git status exited with code {result.exit_code}.',
                content=render_process_output(result),
                metadata=metadata,
            )
        content = result.stdout.rstrip() or 'Working tree clean.'
        return ToolResult.ok(
            'Read Git working tree status.',
            content=content,
            metadata=metadata,
        )


class GitDiffInput(ToolInput):
    staged: bool = False
    path: str | None = Field(default=None, min_length=1)


class GitDiffTool(Tool[GitDiffInput]):
    name = 'git_diff'
    description = (
        'Show unstaged or staged Git changes, optionally limited to one '
        'repository path.'
    )
    input_model = GitDiffInput

    async def execute(self, arguments: GitDiffInput) -> ToolResult:
        command = ['git', 'diff', '--no-ext-diff']
        if arguments.staged:
            command.append('--cached')
        shown_path: str | None = None
        if arguments.path is not None:
            resolved = resolve_repository_path(
                self.root,
                arguments.path,
                must_exist=False,
            )
            shown_path = display_path(self.root, resolved)
            command.extend(['--', shown_path])

        result = await run_process(
            command,
            cwd=self.root,
            timeout_seconds=30,
        )
        metadata = {
            **process_metadata(result),
            'staged': arguments.staged,
            'path': shown_path,
        }
        if result.exit_code != 0:
            return ToolResult.fail(
                'git_diff_failed',
                f'git diff exited with code {result.exit_code}.',
                content=render_process_output(result),
                metadata=metadata,
            )
        content = result.stdout.rstrip()
        return ToolResult.ok(
            'Read Git diff.' if content else 'No matching Git diff.',
            content=content,
            metadata=metadata,
        )
