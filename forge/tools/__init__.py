'''Built-in ForgeCode tools.'''

from pathlib import Path

from forge.tools.base import ToolRegistry
from forge.tools.filesystem import ListDirectoryTool, ReadFileTool
from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.patch import ApplyPatchTool
from forge.tools.search import FindFilesTool, GrepTool
from forge.tools.shell import RunCommandTool


def create_default_registry(root: Path) -> ToolRegistry:
    '''Create the deterministic M1.3 built-in tool set.'''
    return ToolRegistry(
        [
            ListDirectoryTool(root),
            FindFilesTool(root),
            ReadFileTool(root),
            GrepTool(root),
            ApplyPatchTool(root),
            RunCommandTool(root),
            GitStatusTool(root),
            GitDiffTool(root),
        ]
    )


__all__ = [
    'ApplyPatchTool',
    'FindFilesTool',
    'GitDiffTool',
    'GitStatusTool',
    'GrepTool',
    'ListDirectoryTool',
    'ReadFileTool',
    'RunCommandTool',
    'ToolRegistry',
    'create_default_registry',
]
