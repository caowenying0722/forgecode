'''Stdio MCP server exposing public web search and URL fetch tools.'''

from __future__ import annotations

from html.parser import HTMLParser
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


MAX_BYTES = 500_000
MAX_RESULTS = 10
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/126.0 Safari/537.36 ForgeCode-MCP-WebSearch/0.1.0'
)


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
                    'name': 'forge-web-search',
                    'version': '0.1.0',
                },
            }
        elif method == 'tools/list':
            result = {'tools': tool_definitions()}
        elif method == 'tools/call':
            result = call_tool(message.get('params', {}))
        else:
            return error_response(request_id, -32601, f'Unknown method: {method}')
    except Exception as error:
        return error_response(request_id, -32000, str(error))
    return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            'name': 'search_web',
            'description': (
                'Search the public web for current information and return '
                'ranked result titles, URLs, and snippets.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query in any language.',
                    },
                    'max_results': {
                        'type': 'integer',
                        'default': 5,
                        'minimum': 1,
                        'maximum': MAX_RESULTS,
                    },
                },
                'required': ['query'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'fetch_url',
            'description': 'Fetch one public http(s) URL and return text.',
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
        },
        {
            'name': 'get_weather',
            'description': (
                'Get current weather and short forecast for a location using '
                'a public internet weather endpoint.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'City or place name, for example Xiamen.',
                    },
                },
                'required': ['location'],
                'additionalProperties': False,
            },
        },
    ]


def call_tool(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError('params must be an object.')
    name = str(params.get('name', '')).strip()
    arguments = params.get('arguments', {})
    if not isinstance(arguments, dict):
        raise ValueError('arguments must be an object.')
    if name == 'search_web':
        query = str(arguments.get('query', '')).strip()
        max_results = int(arguments.get('max_results', 5))
        results = search_web(query, max_results)
        return text_result(
            json.dumps(
                {'query': query, 'results': results},
                ensure_ascii=False,
                indent=2,
            )
        )
    if name == 'fetch_url':
        url = str(arguments.get('url', '')).strip()
        timeout = float(arguments.get('timeout_seconds', 20))
        text, metadata = fetch_url(url, timeout)
        return text_result(
            json.dumps(metadata, ensure_ascii=False) + '\n\n' + text
        )
    if name == 'get_weather':
        location = str(arguments.get('location', '')).strip()
        return text_result(
            json.dumps(get_weather(location), ensure_ascii=False, indent=2)
        )
    raise ValueError(f'Unknown tool: {name}')


def text_result(text: str) -> dict[str, Any]:
    return {
        'content': [{'type': 'text', 'text': text}],
        'isError': False,
    }


def search_web(query: str, max_results: int) -> list[dict[str, str]]:
    if not query:
        raise ValueError('query is required.')
    if max_results < 1 or max_results > MAX_RESULTS:
        raise ValueError(f'max_results must be in [1, {MAX_RESULTS}].')
    try:
        results = search_bing_rss(query, max_results)
    except Exception:
        results = []
    if len(results) < max_results:
        seen = {item['url'] for item in results}
        for item in search_duckduckgo_html(query, max_results):
            if item['url'] not in seen:
                results.append(item)
                seen.add(item['url'])
            if len(results) >= max_results:
                break
    if not results:
        raise ValueError('No search results returned.')
    return results[:max_results]


def search_bing_rss(query: str, max_results: int) -> list[dict[str, str]]:
    url = f'https://www.bing.com/search?q={quote_plus(query)}&format=rss'
    text, _ = fetch_url(url, 20)
    root = ElementTree.fromstring(text)
    items: list[dict[str, str]] = []
    for item in root.findall('.//item'):
        title = text_of(item, 'title')
        link = text_of(item, 'link')
        snippet = text_of(item, 'description')
        published = text_of(item, 'pubDate')
        if title and link:
            items.append(
                {
                    'title': title,
                    'url': link,
                    'snippet': snippet,
                    'published': published,
                    'source': 'bing-rss',
                }
            )
        if len(items) >= max_results:
            break
    return items


def text_of(item: ElementTree.Element, tag: str) -> str:
    value = item.findtext(tag) or ''
    return ' '.join(value.split())


def search_duckduckgo_html(
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    url = f'https://html.duckduckgo.com/html/?q={quote_plus(query)}'
    text, _ = fetch_url(url, 20)
    parser = DuckDuckGoParser(max_results)
    parser.feed(text)
    return parser.results


class DuckDuckGoParser(HTMLParser):
    def __init__(self, max_results: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_url = ''
        self._last_result_index: int | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attr = {key: value or '' for key, value in attrs}
        classes = set(attr.get('class', '').split())
        if tag == 'a' and 'result__a' in classes:
            self._pending_url = normalize_duckduckgo_url(attr.get('href', ''))
            self._title_parts = []
            self._capture = 'title'
        elif 'result__snippet' in classes:
            self._snippet_parts = []
            self._capture = 'snippet'

    def handle_endtag(self, tag: str) -> None:
        if self._capture == 'title' and tag == 'a':
            title = ' '.join(''.join(self._title_parts).split())
            if title and self._pending_url:
                self.results.append(
                    {
                        'title': title,
                        'url': self._pending_url,
                        'snippet': '',
                        'published': '',
                        'source': 'duckduckgo-html',
                    }
                )
                self._last_result_index = len(self.results) - 1
            self._capture = None
        elif self._capture == 'snippet' and tag in {'a', 'div'}:
            snippet = ' '.join(''.join(self._snippet_parts).split())
            if snippet and self._last_result_index is not None:
                self.results[self._last_result_index]['snippet'] = snippet
            self._capture = None
        if len(self.results) >= self.max_results:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture == 'title':
            self._title_parts.append(data)
        elif self._capture == 'snippet':
            self._snippet_parts.append(data)


def normalize_duckduckgo_url(raw: str) -> str:
    if raw.startswith('//'):
        raw = 'https:' + raw
    parsed = urlparse(raw)
    if parsed.netloc.endswith('duckduckgo.com') and parsed.path == '/l/':
        target = parse_qs(parsed.query).get('uddg', [''])[0]
        return unquote(target) or raw
    return raw


def get_weather(location: str) -> dict[str, Any]:
    if not location:
        raise ValueError('location is required.')
    url = f'https://wttr.in/{quote(location)}?format=j1'
    text, metadata = fetch_url(url, 20)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('weather response must be an object.')
    current = first_mapping(data.get('current_condition'))
    nearest = first_mapping(data.get('nearest_area'))
    forecast = []
    for item in list(data.get('weather') or [])[:3]:
        if not isinstance(item, dict):
            continue
        forecast.append(
            {
                'date': item.get('date', ''),
                'avgtemp_c': item.get('avgtempC', ''),
                'maxtemp_c': item.get('maxtempC', ''),
                'mintemp_c': item.get('mintempC', ''),
                'sun_hour': item.get('sunHour', ''),
                'hourly': [
                    {
                        'time': hour.get('time', ''),
                        'temp_c': hour.get('tempC', ''),
                        'feels_like_c': hour.get('FeelsLikeC', ''),
                        'chance_of_rain': hour.get('chanceofrain', ''),
                        'weather': weather_desc(hour),
                        'wind_kmph': hour.get('windspeedKmph', ''),
                    }
                    for hour in item.get('hourly', [])
                    if isinstance(hour, dict)
                ][:8],
            }
        )
    return {
        'location_query': location,
        'resolved_area': area_name(nearest),
        'current': {
            'observed_at': current.get('localObsDateTime', ''),
            'temp_c': current.get('temp_C', ''),
            'feels_like_c': current.get('FeelsLikeC', ''),
            'humidity': current.get('humidity', ''),
            'weather': weather_desc(current),
            'wind_kmph': current.get('windspeedKmph', ''),
            'precip_mm': current.get('precipMM', ''),
        },
        'forecast': forecast,
        'source': metadata['url'],
    }


def first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def area_name(area: dict[str, Any]) -> str:
    parts = []
    for key in ('areaName', 'region', 'country'):
        value = first_value(area.get(key))
        if value:
            parts.append(value)
    return ', '.join(parts)


def first_value(value: Any) -> str:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get('value', '')).strip()
    return ''


def weather_desc(item: dict[str, Any]) -> str:
    return first_value(item.get('weatherDesc'))


def fetch_url(url: str, timeout_seconds: float) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('Only absolute http(s) URLs are supported.')
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError('timeout_seconds must be in (0, 60].')
    request = Request(
        url,
        headers={
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,text/plain,application/json,application/rss+xml,*/*',
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
