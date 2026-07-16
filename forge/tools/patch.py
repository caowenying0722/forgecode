'''Unified-diff patch application tool.'''

from __future__ import annotations

from pydantic import Field

from forge.tools.base import Tool, ToolInput, ToolResult
from forge.tools.shell import (
    process_metadata,
    render_process_output,
    run_process,
)


class ApplyPatchInput(ToolInput):
    patch: str = Field(min_length=1)


class ApplyPatchTool(Tool[ApplyPatchInput]):
    name = 'apply_patch'
    description = (
        'Apply a unified diff to repository files after validating it with '
        'git apply --check.'
    )
    input_model = ApplyPatchInput

    async def execute(self, arguments: ApplyPatchInput) -> ToolResult:
        check = await run_process(
            ['git', 'apply', '--check', '--whitespace=nowarn', '-'],
            cwd=self.root,
            timeout_seconds=30,
            input_text=arguments.patch,
        )
        if check.exit_code != 0:
            return ToolResult.fail(
                'patch_rejected',
                'Patch validation failed.',
                content=render_process_output(check),
                metadata={'check': process_metadata(check)},
            )

        applied = await run_process(
            ['git', 'apply', '--whitespace=nowarn', '-'],
            cwd=self.root,
            timeout_seconds=30,
            input_text=arguments.patch,
        )
        if applied.exit_code != 0:
            return ToolResult.fail(
                'patch_apply_failed',
                'Patch could not be applied after validation.',
                content=render_process_output(applied),
                metadata={
                    'check': process_metadata(check),
                    'apply': process_metadata(applied),
                },
            )

        status = await run_process(
            ['git', 'status', '--short'],
            cwd=self.root,
            timeout_seconds=30,
        )
        status_text = status.stdout.rstrip()
        changed_files = [
            line[3:].strip()
            for line in status.stdout.splitlines()
            if len(line) >= 4
        ]
        return ToolResult.ok(
            f'Applied patch; {len(changed_files)} working tree paths changed.',
            content=status_text,
            metadata={
                'changed_files': changed_files,
                'check': process_metadata(check),
                'apply': process_metadata(applied),
                'status': process_metadata(status),
            },
        )
