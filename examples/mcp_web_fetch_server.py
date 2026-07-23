'''Example stdio MCP server exposing a simple URL fetch tool.'''

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_BYTES = 500_000


def main() -> None:
    while True:
        message = read_message()
        if message is None:
            return
        response = handle_message(message)
        if response is not None:
            write_message(response)


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get('method')
    request_id = message.get('id')
    if request_id is None:
        return None
    try:
        if method == 'initialize':
            result = {
                'protocolVersion': '2024-11-05',
                'capabilities': {'tools': {}},
                'serverInfo': {
                    'name': 'forge-web-fetch-example',
                    'version': '0.1.0',
                },
            }
        elif method == 'tools/list':
            result = {
                'tools': [
                    {
                        'name': 'fetch_url',
                        'description': (
                            'Fetch one public http(s) URL and return text.'
                        ),
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'url': {
                                    'type': 'string',
                                    'description': 'Absolute http(s) URL.',
                                },
                                'timeout_seconds': {
                                    'type': 'number',
                                    'default': 20,
                                    'minimum': 1,
                                    'maximum': 60,
                                },
                            },
                            'required': ['url'],
                            'additionalProperties': False,
                        },
                    }
                ]
            }
        elif method == 'tools/call':
            params = message.get('params', {})
            result = call_tool(params)
        else:
            return error_response(request_id, -32601, f'Unknown method: {method}')
    except Exception as error:
        return error_response(request_id, -32000, str(error))
    return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def call_tool(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict) or params.get('name') != 'fetch_url':
        raise ValueError('Unknown tool.')
    arguments = params.get('arguments', {})
    if not isinstance(arguments, dict):
        raise ValueError('arguments must be an object.')
    url = str(arguments.get('url', '')).strip()
    timeout = float(arguments.get('timeout_seconds', 20))
    text, metadata = fetch_url(url, timeout)
    return {
        'content': [
            {
                'type': 'text',
                'text': json.dumps(metadata, ensure_ascii=False) + '\n\n' + text,
            }
        ],
        'isError': False,
    }


def fetch_url(url: str, timeout_seconds: float) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('Only absolute http(s) URLs are supported.')
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError('timeout_seconds must be in (0, 60].')
    request = Request(
        url,
        headers={
            'User-Agent': 'ForgeCode-MCP-WebFetch/0.1.0',
            'Accept': 'text/html,text/plain,application/json,*/*',
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get('content-type', '')
            data = response.read(MAX_BYTES + 1)
    except HTTPError as error:
        raise ValueError(f'HTTP request failed with status {error.code}.') from error
    except URLError as error:
        raise ValueError(f'Network request failed: {error.reason}') from error
    truncated = len(data) > MAX_BYTES
    text = data[:MAX_BYTES].decode('utf-8', errors='replace')
    return text, {
        'url': url,
        'content_type': content_type,
        'bytes': min(len(data), MAX_BYTES),
        'truncated': truncated,
    }


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b'\r\n', b'\n'):
            break
        text = line.decode('ascii', errors='replace').strip()
        key, separator, value = text.partition(':')
        if separator:
            headers[key.casefold()] = value.strip()
    length = int(headers['content-length'])
    body = sys.stdin.buffer.read(length)
    message = json.loads(body.decode('utf-8'))
    if not isinstance(message, dict):
        raise ValueError('JSON-RPC message must be an object.')
    return message


def write_message(message: dict[str, Any]) -> None:
    data = json.dumps(
        message,
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    sys.stdout.buffer.write(f'Content-Length: {len(data)}\r\n\r\n'.encode('ascii'))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def error_response(
    request_id: Any,
    code: int,
    message: str,
) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'error': {
            'code': code,
            'message': message,
        },
    }


if __name__ == '__main__':
    main()
