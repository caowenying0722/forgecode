'''Minimal MCP host/client support over stdio transport.'''

from __future__ import annotations

import atexit
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any


JSONRPC_VERSION = '2.0'
DEFAULT_TIMEOUT_SECONDS = 30.0
SUPPORTED_PROTOCOL_VERSION = '2024-11-05'


class MCPConfigurationError(ValueError):
    '''Raised when MCP server configuration is invalid.'''


class MCPProtocolError(RuntimeError):
    '''Raised when an MCP server violates the JSON-RPC protocol.'''


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    env: dict[str, str] | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class MCPRemoteTool:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    client: StdioMCPClient


class StdioMCPClient:
    '''Synchronous MCP stdio client using Content-Length JSON-RPC frames.'''

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._next_id = 1
        self._reader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f'mcp-{config.name}-reader',
        )
        self._process = subprocess.Popen(
            [config.command, *config.args],
            cwd=str(config.cwd) if config.cwd is not None else None,
            env=merged_env(config.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.close)
        self._initialize()

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()
        self._reader.shutdown(wait=False, cancel_futures=True)

    def list_tools(self) -> tuple[MCPRemoteTool, ...]:
        response = self.request('tools/list', {})
        tools = response.get('tools', [])
        if not isinstance(tools, list):
            raise MCPProtocolError('MCP tools/list result must contain tools list.')
        remote_tools: list[MCPRemoteTool] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            if not name:
                continue
            schema = item.get('inputSchema', {})
            if not isinstance(schema, dict):
                schema = {}
            description = str(item.get('description', '')).strip()
            remote_tools.append(
                MCPRemoteTool(
                    server_name=self.config.name,
                    name=name,
                    description=description,
                    input_schema=schema,
                    client=self,
                )
            )
        return tuple(remote_tools)

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.request(
            'tools/call',
            {
                'name': name,
                'arguments': dict(arguments),
            },
        )

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._write_message(
                {
                    'jsonrpc': JSONRPC_VERSION,
                    'id': request_id,
                    'method': method,
                    'params': dict(params or {}),
                }
            )
            future = self._reader.submit(self._read_response_for_id, request_id)
            try:
                message = future.result(timeout=self.config.timeout_seconds)
            except TimeoutError as error:
                self.close()
                raise MCPProtocolError(
                    f'MCP {method} timed out after '
                    f'{self.config.timeout_seconds:g}s.'
                ) from error
            if 'error' in message:
                error = message['error']
                if isinstance(error, dict):
                    detail = error.get('message', error)
                else:
                    detail = error
                raise MCPProtocolError(f'MCP {method} failed: {detail}')
            result = message.get('result', {})
            if not isinstance(result, dict):
                raise MCPProtocolError(
                    f'MCP {method} result must be an object.'
                )
            return result

    def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._write_message(
                {
                    'jsonrpc': JSONRPC_VERSION,
                    'method': method,
                    'params': dict(params or {}),
                }
            )

    def _initialize(self) -> None:
        self.request(
            'initialize',
            {
                'protocolVersion': SUPPORTED_PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': {
                    'name': 'ForgeCode',
                    'version': '0.1.0',
                },
            },
        )
        self.notify('notifications/initialized', {})

    def _write_message(self, payload: Mapping[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            raise MCPProtocolError(
                f'MCP server {self.config.name!r} is not running.'
            )
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8')
        header = f'Content-Length: {len(data)}\r\n\r\n'.encode('ascii')
        stdin.write(header + data)
        stdin.flush()

    def _read_response_for_id(self, request_id: int) -> dict[str, Any]:
        while True:
            message = self._read_message()
            if message.get('id') == request_id:
                return message

    def _read_message(self) -> dict[str, Any]:
        stdout = self._process.stdout
        if stdout is None:
            raise MCPProtocolError(
                f'MCP server {self.config.name!r} has no stdout pipe.'
            )
        headers: dict[str, str] = {}
        while True:
            line = stdout.readline()
            if not line:
                raise MCPProtocolError(
                    f'MCP server {self.config.name!r} closed stdout.'
                )
            if line in (b'\r\n', b'\n'):
                break
            text = line.decode('ascii', errors='replace').strip()
            key, separator, value = text.partition(':')
            if separator:
                headers[key.casefold()] = value.strip()
        try:
            length = int(headers['content-length'])
        except (KeyError, ValueError) as error:
            raise MCPProtocolError('MCP frame missing valid Content-Length.') from error
        body = stdout.read(length)
        if len(body) != length:
            raise MCPProtocolError('MCP frame ended before Content-Length bytes.')
        try:
            message = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as error:
            raise MCPProtocolError('MCP frame body is not valid JSON.') from error
        if not isinstance(message, dict):
            raise MCPProtocolError('MCP frame body must be a JSON object.')
        return message


class MCPClientManager:
    '''Own configured MCP clients and expose their remote tools.'''

    def __init__(self, clients: Iterable[StdioMCPClient]) -> None:
        self.clients = tuple(clients)

    @classmethod
    def from_config_file(cls, root: Path) -> MCPClientManager:
        path = root / '.forge' / 'mcp.json'
        if not path.is_file():
            return cls(())
        data = json.loads(path.read_text(encoding='utf-8'))
        configs = parse_mcp_config(data, root)
        clients = [StdioMCPClient(config) for config in configs]
        return cls(clients)

    def list_tools(self) -> tuple[MCPRemoteTool, ...]:
        tools: list[MCPRemoteTool] = []
        for client in self.clients:
            tools.extend(client.list_tools())
        return tuple(tools)

    def close(self) -> None:
        for client in self.clients:
            client.close()


def parse_mcp_config(data: Any, root: Path) -> tuple[MCPServerConfig, ...]:
    if not isinstance(data, dict):
        raise MCPConfigurationError('.forge/mcp.json must contain a JSON object.')
    raw_servers = data.get('servers', {})
    if not isinstance(raw_servers, dict):
        raise MCPConfigurationError('mcp.json field `servers` must be an object.')
    configs: list[MCPServerConfig] = []
    for name, raw in raw_servers.items():
        server_name = sanitize_server_name(str(name))
        if not isinstance(raw, dict):
            raise MCPConfigurationError(f'MCP server {name!r} must be an object.')
        transport = str(raw.get('transport', 'stdio'))
        if transport != 'stdio':
            raise MCPConfigurationError(
                f'MCP server {name!r} uses unsupported transport {transport!r}.'
            )
        command = str(raw.get('command', '')).strip()
        if not command:
            raise MCPConfigurationError(
                f'MCP server {name!r} must configure `command`.'
            )
        raw_args = raw.get('args', [])
        if not isinstance(raw_args, list) or not all(
            isinstance(item, str) for item in raw_args
        ):
            raise MCPConfigurationError(
                f'MCP server {name!r} field `args` must be a string list.'
            )
        raw_env = raw.get('env', None)
        if raw_env is not None and (
            not isinstance(raw_env, dict)
            or not all(isinstance(key, str) for key in raw_env)
        ):
            raise MCPConfigurationError(
                f'MCP server {name!r} field `env` must be an object.'
            )
        raw_cwd = raw.get('cwd', None)
        cwd = resolve_mcp_cwd(root, raw_cwd, name=str(name))
        raw_timeout = raw.get('timeout_seconds', DEFAULT_TIMEOUT_SECONDS)
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as error:
            raise MCPConfigurationError(
                f'MCP server {name!r} timeout_seconds must be a number.'
            ) from error
        if timeout <= 0 or timeout > 300:
            raise MCPConfigurationError(
                f'MCP server {name!r} timeout_seconds must be in (0, 300].'
            )
        configs.append(
            MCPServerConfig(
                name=server_name,
                command=command,
                args=tuple(raw_args),
                cwd=cwd,
                env={key: str(value) for key, value in (raw_env or {}).items()},
                timeout_seconds=timeout,
            )
        )
    return tuple(configs)


def sanitize_server_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', name.strip())
    return sanitized.strip('_') or 'server'


def resolve_mcp_cwd(root: Path, raw_cwd: Any, *, name: str) -> Path | None:
    if raw_cwd is None:
        return root
    if not isinstance(raw_cwd, str):
        raise MCPConfigurationError(
            f'MCP server {name!r} field `cwd` must be a string.'
        )
    candidate = Path(raw_cwd)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def merged_env(extra: Mapping[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra or {})
    return env
