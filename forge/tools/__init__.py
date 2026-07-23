'''Built-in ForgeCode tools.'''

from pathlib import Path

from forge.tools.base import ToolRegistry
from forge.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    ReplaceTextTool,
    WriteFileChunkTool,
    WriteFileTool,
)
from forge.tools.finish import FinishTaskTool
from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.mcp import MCPTool
from forge.tools.patch import ApplyPatchTool
from forge.tools.search import FindFilesTool, GrepTool
from forge.tools.shell import RunCommandTool
from forge.tools.verify import VerifyTool
from forge.runtime.workspace import WorkspaceTracker
from forge.mcp import MCPClientManager


def create_default_registry(root: Path) -> ToolRegistry:
    '''Create built-in tools sharing one task-local workspace tracker.'''
    tracker = WorkspaceTracker(root)
    mcp_manager = MCPClientManager.from_config_file(root)
    mcp_tools = [
        MCPTool(root, remote_tool)
        for remote_tool in mcp_manager.list_tools()
    ]
    return ToolRegistry(
        [
            ListDirectoryTool(root),
            FindFilesTool(root),
            ReadFileTool(root),
            GrepTool(root),
            WriteFileTool(root),
            WriteFileChunkTool(root),
            ReplaceTextTool(root),
            ApplyPatchTool(root),
            RunCommandTool(root),
            VerifyTool(root, tracker),
            GitStatusTool(root),
            GitDiffTool(root),
            *mcp_tools,
            FinishTaskTool(root),
        ],
        workspace_tracker=tracker,
    )


__all__ = [
    'ApplyPatchTool',
    'FindFilesTool',
    'FinishTaskTool',
    'GitDiffTool',
    'GitStatusTool',
    'GrepTool',
    'ListDirectoryTool',
    'MCPTool',
    'ReadFileTool',
    'ReplaceTextTool',
    'RunCommandTool',
    'VerifyTool',
    'WriteFileChunkTool',
    'WriteFileTool',
    'ToolRegistry',
    'create_default_registry',
]
