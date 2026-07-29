'''Tests for stdio MCP integration.'''

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from forge.mcp.client import MCPClientManager, parse_mcp_config
from forge.mcp.client import MCPRemoteTool
from forge.runtime.agent_loop import Conversation
from forge.tools import create_default_registry
from forge.tools.base import ToolRegistry
from forge.tools.mcp import MCPTool, infer_mcp_tool_effect


SERVER_SCRIPT = r'''
import json
import sys

def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b'\r\n', b'\n'):
            break
        key, _, value = line.decode('ascii').strip().partition(':')
        headers[key.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers['content-length'])))

def write_message(message):
    data = json.dumps(message, separators=(',', ':')).encode()
    sys.stdout.buffer.write(f'Content-Length: {len(data)}\r\n\r\n'.encode())
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

while True:
    message = read_message()
    if message is None:
        break
    method = message.get('method')
    if 'id' not in message:
        continue
    if method == 'initialize':
        result = {
            'protocolVersion': '2024-11-05',
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'demo', 'version': '1'},
        }
    elif method == 'tools/list':
        result = {
            'tools': [{
                'name': 'echo',
                'description': 'Echo text.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                },
            }]
        }
    elif method == 'tools/call':
        text = message['params']['arguments']['text']
        result = {'content': [{'type': 'text', 'text': 'echo:' + text}]}
    else:
        result = {}
    write_message({'jsonrpc': '2.0', 'id': message['id'], 'result': result})
'''


class DummyMCPClient:
    pass


def test_mcp_config_parses_stdio_servers(tmp_path: Path) -> None:
    configs = parse_mcp_config(
        {
            'servers': {
                'demo-server': {
                    'command': sys.executable,
                    'args': ['server.py'],
                    'cwd': '.',
                }
            }
        },
        tmp_path,
    )

    assert len(configs) == 1
    assert configs[0].name == 'demo_server'
    assert configs[0].command == sys.executable
    assert configs[0].args == ('server.py',)
    assert configs[0].cwd == tmp_path


def test_mcp_manager_lists_and_calls_stdio_tool(tmp_path: Path) -> None:
    server = tmp_path / 'server.py'
    server.write_text(SERVER_SCRIPT, encoding='utf-8')
    manager = MCPClientManager.from_config_file(
        write_config(tmp_path, server),
        app_root=tmp_path / 'empty-app',
    )
    try:
        tools = manager.list_tools()
        result = tools[0].client.call_tool('echo', {'text': 'hello'})
    finally:
        manager.close()

    assert len(tools) == 1
    assert tools[0].server_name == 'demo'
    assert tools[0].name == 'echo'
    assert result['content'][0]['text'] == 'echo:hello'


def test_default_registry_exposes_mcp_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    server = tmp_path / 'server.py'
    server.write_text(SERVER_SCRIPT, encoding='utf-8')
    write_config(tmp_path, server)
    monkeypatch.setattr(
        'forge.mcp.client.forge_app_root',
        lambda *, app_root=None: tmp_path / 'empty-app',
    )

    registry = create_default_registry(tmp_path)
    result = asyncio.run(
        registry.execute('mcp_demo_echo', {'text': 'hello'})
    )

    assert 'mcp_demo_echo' in registry.names
    assert result.success is True
    assert result.content == 'echo:hello'
    assert result.metadata['mcp_server'] == 'demo'
    assert result.metadata['mcp_tool'] == 'echo'


def test_fetch_url_mcp_tool_is_read_only(tmp_path: Path) -> None:
    remote = MCPRemoteTool(
        server_name='web',
        name='fetch_url',
        description='Fetch one public http(s) URL and return text.',
        input_schema={'type': 'object'},
        client=DummyMCPClient(),  # type: ignore[arg-type]
    )
    tool = MCPTool(tmp_path, remote)

    assert infer_mcp_tool_effect(remote) == 'read_only'
    assert tool.effect == 'read_only'


def test_read_only_tool_surface_includes_read_only_mcp_tool(
    tmp_path: Path,
) -> None:
    remote = MCPRemoteTool(
        server_name='web',
        name='fetch_url',
        description='Fetch one public http(s) URL and return text.',
        input_schema={'type': 'object'},
        client=DummyMCPClient(),  # type: ignore[arg-type]
    )
    conversation = Conversation(
        client=object(),  # type: ignore[arg-type]
        registry=ToolRegistry([MCPTool(tmp_path, remote)]),
    )

    names = {
        str(definition.get('name'))
        for definition in conversation._plan_mode_tools() or ()
    }

    assert 'mcp_web_fetch_url' in names


def test_project_mcp_config_overrides_user_server_with_same_name(
    tmp_path: Path,
) -> None:
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    (project / '.forge').mkdir(parents=True)
    home.mkdir()
    server = tmp_path / 'server.py'
    server.write_text(SERVER_SCRIPT, encoding='utf-8')
    (home / 'mcp.json').write_text(
        json.dumps(
            {
                'servers': {
                    'demo': {
                        'command': sys.executable,
                        'args': [str(server)],
                    }
                }
            }
        ),
        encoding='utf-8',
    )
    (project / '.forge' / 'mcp.json').write_text(
        json.dumps(
            {
                'servers': {
                    'demo': {
                        'command': sys.executable,
                        'args': [str(server)],
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    manager = MCPClientManager.from_config_file(
        project,
        home=home,
        app_root=tmp_path / 'empty-app',
    )

    assert len(manager.clients) == 1
    assert manager.clients[0].config.cwd == project.resolve()
    manager.close()


def test_app_root_mcp_config_is_used_as_workspace_fallback(
    tmp_path: Path,
) -> None:
    app_root = tmp_path / 'app'
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    app_root.mkdir()
    home.mkdir()
    project.mkdir()
    server = tmp_path / 'server.py'
    server.write_text(SERVER_SCRIPT, encoding='utf-8')
    write_config(app_root, server)

    manager = MCPClientManager.from_config_file(
        project,
        home=home,
        app_root=app_root,
    )
    try:
        tools = manager.list_tools()
    finally:
        manager.close()

    assert len(tools) == 1
    assert tools[0].server_name == 'demo'
    assert manager.clients[0].config.cwd == app_root.resolve()


def test_project_mcp_config_overrides_app_root_fallback(
    tmp_path: Path,
) -> None:
    app_root = tmp_path / 'app'
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    app_root.mkdir()
    home.mkdir()
    project.mkdir()
    server = tmp_path / 'server.py'
    server.write_text(SERVER_SCRIPT, encoding='utf-8')
    write_config(app_root, server)
    write_config(project, server)

    manager = MCPClientManager.from_config_file(
        project,
        home=home,
        app_root=app_root,
    )

    assert len(manager.clients) == 1
    assert manager.clients[0].config.cwd == project.resolve()
    manager.close()


def write_config(tmp_path: Path, server: Path) -> Path:
    config_dir = tmp_path / '.forge'
    config_dir.mkdir()
    (config_dir / 'mcp.json').write_text(
        json.dumps(
            {
                'servers': {
                    'demo': {
                        'transport': 'stdio',
                        'command': sys.executable,
                        'args': [str(server)],
                        'cwd': str(tmp_path),
                    }
                }
            }
        ),
        encoding='utf-8',
    )
    return tmp_path
