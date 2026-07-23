'''Adapter from MCP remote tools into ForgeCode's Tool interface.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Any, Mapping

from forge.mcp.client import MCPProtocolError, MCPRemoteTool
from forge.tools.base import Tool, ToolInput, ToolResult


class MCPTool(Tool[ToolInput]):
    '''Expose one MCP remote tool through ForgeCode's registry.'''

    input_model = ToolInput
    effect = 'process'

    def __init__(self, root: Path, remote: MCPRemoteTool) -> None:
        super().__init__(root)
        self.remote = remote
        self.name = mcp_tool_name(remote.server_name, remote.name)
        self.description = (
            f'MCP tool `{remote.name}` from server `{remote.server_name}`. '
            + (remote.description or 'Use according to its input schema.')
        )

    @property
    def definition(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': normalize_input_schema(self.remote.input_schema),
        }

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            result = await asyncio.to_thread(
                self.remote.client.call_tool,
                self.remote.name,
                arguments,
            )
        except MCPProtocolError as error:
            return ToolResult.fail(
                'mcp_protocol_error',
                str(error),
                metadata=self._metadata(),
            )
        except Exception as error:
            return ToolResult.fail(
                'mcp_tool_failed',
                f'MCP tool {self.remote.name} failed: {error}',
                details={'exception_type': type(error).__name__},
                metadata=self._metadata(),
            )
        return mcp_call_result_to_tool_result(result, self._metadata())

    async def execute(self, arguments: ToolInput) -> ToolResult:
        raise NotImplementedError('MCPTool overrides run directly.')

    def _metadata(self) -> dict[str, Any]:
        return {
            'mcp_server': self.remote.server_name,
            'mcp_tool': self.remote.name,
        }


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    raw = f'mcp_{server_name}_{tool_name}'
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', raw)
    return sanitized.strip('_')[:64] or 'mcp_tool'


def normalize_input_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(schema)
    normalized.setdefault('type', 'object')
    normalized.setdefault('properties', {})
    if not isinstance(normalized.get('properties'), dict):
        normalized['properties'] = {}
    return normalized


def mcp_call_result_to_tool_result(
    result: Mapping[str, Any],
    metadata: dict[str, Any],
) -> ToolResult:
    content = render_mcp_content(result.get('content', []))
    is_error = bool(result.get('isError', False))
    merged_metadata = {**metadata, 'mcp_is_error': is_error}
    if is_error:
        return ToolResult.fail(
            'mcp_tool_error',
            'MCP tool returned an error result.',
            content=content,
            metadata=merged_metadata,
        )
    return ToolResult.ok(
        'MCP tool completed.',
        content=content,
        metadata=merged_metadata,
    )


def render_mcp_content(content: Any) -> str:
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, default=str)
    rendered: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            rendered.append(str(block))
            continue
        block_type = block.get('type')
        if block_type == 'text':
            rendered.append(str(block.get('text', '')))
        else:
            rendered.append(json.dumps(block, ensure_ascii=False, default=str))
    return '\n'.join(item for item in rendered if item)
