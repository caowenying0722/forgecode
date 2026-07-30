'''Environment-backed ForgeCode configuration.'''

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

from dotenv import dotenv_values


DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com'
DEFAULT_MODEL_MAX_TOKENS = 8_192
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 120.0
SUPPORTED_MODEL_IDS = (
    'gpt-5.3-codex-spark',
    'gpt-5.4-mini',
    'gpt-5.4',
    'gpt-5.5',
    'gpt-5.6-sol',
)
USER_CONFIG_TEMPLATE = '''# ForgeCode user defaults
[model]
model_id = ""
base_url = "https://api.anthropic.com"
max_tokens = 8192
# context_window = 2000000
request_timeout_seconds = 120
'''
USER_ENV_TEMPLATE = '''# Keep credentials out of config.toml.
ANTHROPIC_API_KEY=
'''

CONFIG_KEYS = {
    'api_key': 'ANTHROPIC_API_KEY',
    'model_id': 'MODEL_ID',
    'base_url': 'ANTHROPIC_BASE_URL',
    'max_tokens': 'MODEL_MAX_TOKENS',
    'context_window': 'MODEL_CONTEXT_WINDOW',
    'request_timeout_seconds': 'MODEL_REQUEST_TIMEOUT_SECONDS',
}


class ConfigurationError(ValueError):
    '''Raised when ForgeCode model configuration is incomplete or invalid.'''


def normalize_supported_model_id(model_id: str) -> str:
    '''Return a supported model id or raise a user-facing config error.'''
    normalized = model_id.strip()
    if normalized not in SUPPORTED_MODEL_IDS:
        choices = ', '.join(SUPPORTED_MODEL_IDS)
        raise ConfigurationError(f'MODEL_ID must be one of: {choices}.')
    return normalized


@dataclass(frozen=True, slots=True)
class ForgeConfig:
    '''Validated configuration used to create the first model client.'''

    api_key: str
    model_id: str
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    max_tokens: int = DEFAULT_MODEL_MAX_TOKENS
    context_window: int | None = None
    request_timeout_seconds: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        model_id = self.model_id.strip()
        base_url = self.base_url.strip().rstrip('/')

        if not api_key:
            raise ConfigurationError('ANTHROPIC_API_KEY is not set.')
        if not model_id:
            raise ConfigurationError('MODEL_ID is not set.')
        if not 1_024 <= self.max_tokens <= 32_768:
            raise ConfigurationError(
                'MODEL_MAX_TOKENS must be between 1024 and 32768.'
            )
        if self.context_window is not None:
            if not 4_096 <= self.context_window <= 2_000_000:
                raise ConfigurationError(
                    'MODEL_CONTEXT_WINDOW must be between 4096 and 2000000.'
                )
            if self.context_window <= self.max_tokens:
                raise ConfigurationError(
                    'MODEL_CONTEXT_WINDOW must be greater than '
                    'MODEL_MAX_TOKENS.'
                )
        if not 10 <= self.request_timeout_seconds <= 600:
            raise ConfigurationError(
                'MODEL_REQUEST_TIMEOUT_SECONDS must be between 10 and 600.'
            )

        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {'http', 'https'} or not parsed_url.netloc:
            raise ConfigurationError(
                'ANTHROPIC_BASE_URL must be an absolute http(s) URL.'
            )

        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'model_id', model_id)
        object.__setattr__(self, 'base_url', base_url)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> ForgeConfig:
        '''Load env > project > user settings with safe default values.'''
        if environ is None:
            resolved_cwd = (cwd or Path.cwd()).resolve()
            resolved_home = forge_home(home=home)
            source: Mapping[str, str] = os.environ
            values = {
                **read_config_file(resolved_home / 'config.toml'),
                **read_dotenv_file(resolved_home / '.env'),
                **read_config_file(resolved_cwd / '.forge' / 'config.toml'),
                **read_dotenv_file(resolved_cwd / '.env'),
                **{
                    key: value
                    for key, value in source.items()
                    if key in CONFIG_KEYS.values()
                },
            }
        else:
            values = dict(environ)

        raw_max_tokens = str(values.get(
            'MODEL_MAX_TOKENS',
            str(DEFAULT_MODEL_MAX_TOKENS),
        ))
        try:
            max_tokens = int(raw_max_tokens)
        except ValueError as error:
            raise ConfigurationError(
                'MODEL_MAX_TOKENS must be an integer.'
            ) from error
        raw_context_window = str(
            values.get('MODEL_CONTEXT_WINDOW', '')
        ).strip()
        try:
            context_window = (
                int(raw_context_window) if raw_context_window else None
            )
        except ValueError as error:
            raise ConfigurationError(
                'MODEL_CONTEXT_WINDOW must be an integer.'
            ) from error
        raw_request_timeout = str(values.get(
            'MODEL_REQUEST_TIMEOUT_SECONDS',
            str(DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS),
        ))
        try:
            request_timeout_seconds = float(raw_request_timeout)
        except ValueError as error:
            raise ConfigurationError(
                'MODEL_REQUEST_TIMEOUT_SECONDS must be a number.'
            ) from error

        return cls(
            api_key=str(values.get('ANTHROPIC_API_KEY', '')),
            model_id=str(values.get('MODEL_ID', '')),
            base_url=str(values.get(
                'ANTHROPIC_BASE_URL',
                DEFAULT_ANTHROPIC_BASE_URL,
            )),
            max_tokens=max_tokens,
            context_window=context_window,
            request_timeout_seconds=request_timeout_seconds,
        )


def forge_home(*, home: Path | None = None) -> Path:
    if home is not None:
        return home.expanduser().resolve()
    configured = os.environ.get('FORGE_HOME', '').strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / '.forge'


def read_config_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        parsed = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f'Invalid ForgeCode config: {path}: {error}') from error
    model = parsed.get('model', parsed)
    if not isinstance(model, dict):
        raise ConfigurationError(f'ForgeCode config [model] must be a table: {path}')
    values: dict[str, object] = {}
    for key, environment_key in CONFIG_KEYS.items():
        if key in model and model[key] is not None:
            values[environment_key] = model[key]
    return values


def read_dotenv_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return {
        key: value
        for key, value in dotenv_values(path).items()
        if key in CONFIG_KEYS.values() and value is not None
    }


def initialize_user_config(*, home: Path | None = None) -> tuple[Path, Path]:
    directory = forge_home(home=home)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.toml'
    env_path = directory / '.env'
    if not config_path.exists():
        config_path.write_text(USER_CONFIG_TEMPLATE, encoding='utf-8')
    if not env_path.exists():
        env_path.write_text(USER_ENV_TEMPLATE, encoding='utf-8')
    return config_path, env_path


def write_user_config(
    config: ForgeConfig,
    *,
    home: Path | None = None,
) -> tuple[Path, Path]:
    '''Persist validated user defaults without printing credential contents.'''
    directory = forge_home(home=home)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.toml'
    env_path = directory / '.env'
    _write_user_model_config(config_path, config)
    env_path.write_text(
        f'ANTHROPIC_API_KEY={dotenv_string(config.api_key)}\n',
        encoding='utf-8',
    )
    if os.name != 'nt':
        env_path.chmod(0o600)
    return config_path, env_path


def update_user_model_id(
    model_id: str,
    *,
    home: Path | None = None,
) -> Path:
    '''Persist the selected model as the user-level default model.'''
    normalized = normalize_supported_model_id(model_id)
    directory = forge_home(home=home)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.toml'
    existing = read_config_file(config_path)
    raw_max_tokens = existing.get(
        'MODEL_MAX_TOKENS',
        DEFAULT_MODEL_MAX_TOKENS,
    )
    raw_context_window = existing.get('MODEL_CONTEXT_WINDOW')
    raw_timeout = existing.get(
        'MODEL_REQUEST_TIMEOUT_SECONDS',
        DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    )
    try:
        max_tokens = int(raw_max_tokens)
        context_window = (
            int(raw_context_window)
            if raw_context_window not in (None, '')
            else None
        )
        request_timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f'Invalid ForgeCode config: {config_path}: {error}'
        ) from error
    config = ForgeConfig(
        api_key='placeholder',
        model_id=normalized,
        base_url=str(
            existing.get('ANTHROPIC_BASE_URL', DEFAULT_ANTHROPIC_BASE_URL)
        ),
        max_tokens=max_tokens,
        context_window=context_window,
        request_timeout_seconds=request_timeout_seconds,
    )
    _write_user_model_config(config_path, config)
    return config_path


def _write_user_model_config(path: Path, config: ForgeConfig) -> None:
    context_line = (
        f'context_window = {config.context_window}\n'
        if config.context_window is not None
        else ''
    )
    path.write_text(
        '# ForgeCode user defaults\n'
        '[model]\n'
        f'model_id = {toml_string(config.model_id)}\n'
        f'base_url = {toml_string(config.base_url)}\n'
        f'max_tokens = {config.max_tokens}\n'
        f'{context_line}'
        f'request_timeout_seconds = {config.request_timeout_seconds:g}\n',
        encoding='utf-8',
    )


def toml_string(value: str) -> str:
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def dotenv_string(value: str) -> str:
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'
