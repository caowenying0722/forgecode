'''Tests for environment-backed ForgeCode configuration.'''

from pathlib import Path

import pytest

from forge.config import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_MODEL_MAX_TOKENS,
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    ConfigurationError,
    ForgeConfig,
    initialize_user_config,
    update_user_model_id,
    write_user_config,
)


def test_config_uses_official_base_url_by_default() -> None:
    config = ForgeConfig.from_env(
        {
            'ANTHROPIC_API_KEY': ' test-key ',
            'MODEL_ID': ' claude-test ',
        }
    )

    assert config.api_key == 'test-key'
    assert config.model_id == 'claude-test'
    assert config.base_url == DEFAULT_ANTHROPIC_BASE_URL
    assert config.max_tokens == DEFAULT_MODEL_MAX_TOKENS
    assert config.context_window is None
    assert (
        config.request_timeout_seconds
        == DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    )


def test_config_accepts_anthropic_compatible_base_url() -> None:
    config = ForgeConfig.from_env(
        {
            'ANTHROPIC_API_KEY': 'test-key',
            'MODEL_ID': 'claude-test',
            'ANTHROPIC_BASE_URL': 'http://localhost:8080/anthropic/',
        }
    )

    assert config.base_url == 'http://localhost:8080/anthropic'


def test_config_loads_dotenv_from_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('MODEL_ID', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    (tmp_path / '.env').write_text(
        'ANTHROPIC_API_KEY=dotenv-key\n'
        'MODEL_ID=dotenv-model\n'
        'ANTHROPIC_BASE_URL=http://localhost:8080/anthropic/\n',
        encoding='utf-8',
    )

    config = ForgeConfig.from_env()

    assert config.api_key == 'dotenv-key'
    assert config.model_id == 'dotenv-model'
    assert config.base_url == 'http://localhost:8080/anthropic'


def test_environment_variables_override_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'environment-key')
    monkeypatch.setenv('MODEL_ID', 'environment-model')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://environment.example.com')
    (tmp_path / '.env').write_text(
        'ANTHROPIC_API_KEY=dotenv-key\n'
        'MODEL_ID=dotenv-model\n'
        'ANTHROPIC_BASE_URL=https://dotenv.example.com\n',
        encoding='utf-8',
    )

    config = ForgeConfig.from_env()

    assert config.api_key == 'environment-key'
    assert config.model_id == 'environment-model'
    assert config.base_url == 'https://environment.example.com'


def test_user_project_and_environment_config_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    (project / '.forge').mkdir(parents=True)
    home.mkdir()
    (home / 'config.toml').write_text(
        '[model]\nmodel_id = "user-model"\nmax_tokens = 4096\n',
        encoding='utf-8',
    )
    (home / '.env').write_text(
        'ANTHROPIC_API_KEY=user-key\n', encoding='utf-8'
    )
    (project / '.forge' / 'config.toml').write_text(
        '[model]\nmodel_id = "project-model"\nmax_tokens = 16384\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('MODEL_ID', 'environment-model')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    config = ForgeConfig.from_env(cwd=project, home=home)

    assert config.api_key == 'user-key'
    assert config.model_id == 'environment-model'
    assert config.max_tokens == 16_384


def test_initialize_user_config_does_not_overwrite_existing_files(
    tmp_path: Path,
) -> None:
    config_path, env_path = initialize_user_config(home=tmp_path)
    config_path.write_text('custom = true\n', encoding='utf-8')
    env_path.write_text('CUSTOM=value\n', encoding='utf-8')

    initialize_user_config(home=tmp_path)

    assert config_path.read_text(encoding='utf-8') == 'custom = true\n'
    assert env_path.read_text(encoding='utf-8') == 'CUSTOM=value\n'


def test_write_user_config_round_trips_without_exposing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ForgeConfig(
        api_key='secret-key',
        model_id='test-model',
        max_tokens=16_384,
        context_window=2_000_000,
    )
    write_user_config(original, home=tmp_path)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('MODEL_ID', raising=False)

    loaded = ForgeConfig.from_env(cwd=tmp_path / 'project', home=tmp_path)

    assert loaded == original


def test_update_user_model_id_updates_global_default_from_any_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    project.mkdir()
    home.mkdir()
    (home / 'config.toml').write_text(
        '[model]\n'
        'model_id = "gpt-5.3-codex-spark"\n'
        'base_url = "http://localhost:63962"\n'
        'max_tokens = 8192\n'
        'context_window = 128000\n'
        'request_timeout_seconds = 120\n',
        encoding='utf-8',
    )
    (home / '.env').write_text(
        'ANTHROPIC_API_KEY=user-key\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('MODEL_ID', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    config_path = update_user_model_id('gpt-5.5', home=home)
    loaded = ForgeConfig.from_env(cwd=project, home=home)

    assert config_path == home / 'config.toml'
    assert loaded.model_id == 'gpt-5.5'
    assert loaded.base_url == 'http://localhost:63962'
    assert loaded.context_window == 128000
    assert 'ANTHROPIC_API_KEY' not in config_path.read_text(encoding='utf-8')


def test_update_user_model_id_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match='MODEL_ID must be one of'):
        update_user_model_id('unknown-model', home=tmp_path)


def test_config_rejects_missing_api_key() -> None:
    with pytest.raises(ConfigurationError, match='ANTHROPIC_API_KEY'):
        ForgeConfig.from_env({})


def test_config_rejects_missing_model_id() -> None:
    with pytest.raises(ConfigurationError, match='MODEL_ID'):
        ForgeConfig.from_env({'ANTHROPIC_API_KEY': 'test-key'})


def test_config_rejects_invalid_base_url() -> None:
    with pytest.raises(ConfigurationError, match='ANTHROPIC_BASE_URL'):
        ForgeConfig(
            api_key='test-key',
            model_id='claude-test',
            base_url='localhost:8080',
        )


def test_config_reads_and_validates_model_max_tokens() -> None:
    config = ForgeConfig.from_env(
        {
            'ANTHROPIC_API_KEY': 'test-key',
            'MODEL_ID': 'test-model',
            'MODEL_MAX_TOKENS': '16384',
        }
    )

    assert config.max_tokens == 16_384


def test_config_reads_model_context_window() -> None:
    config = ForgeConfig.from_env(
        {
            'ANTHROPIC_API_KEY': 'test-key',
            'MODEL_ID': 'test-model',
            'MODEL_MAX_TOKENS': '8192',
            'MODEL_CONTEXT_WINDOW': '128000',
        }
    )

    assert config.context_window == 128_000


def test_config_reads_model_request_timeout() -> None:
    config = ForgeConfig.from_env(
        {
            'ANTHROPIC_API_KEY': 'test-key',
            'MODEL_ID': 'test-model',
            'MODEL_REQUEST_TIMEOUT_SECONDS': '45.5',
        }
    )

    assert config.request_timeout_seconds == 45.5


@pytest.mark.parametrize('value', ['invalid', '9', '601'])
def test_config_rejects_invalid_model_request_timeout(value: str) -> None:
    with pytest.raises(
        ConfigurationError,
        match='MODEL_REQUEST_TIMEOUT_SECONDS',
    ):
        ForgeConfig.from_env(
            {
                'ANTHROPIC_API_KEY': 'test-key',
                'MODEL_ID': 'test-model',
                'MODEL_REQUEST_TIMEOUT_SECONDS': value,
            }
        )


@pytest.mark.parametrize('value', ['invalid', '1000', '40000'])
def test_config_rejects_invalid_model_max_tokens(value: str) -> None:
    with pytest.raises(ConfigurationError, match='MODEL_MAX_TOKENS'):
        ForgeConfig.from_env(
            {
                'ANTHROPIC_API_KEY': 'test-key',
                'MODEL_ID': 'test-model',
                'MODEL_MAX_TOKENS': value,
            }
        )


@pytest.mark.parametrize('value', ['invalid', '4000', '3000000', '8192'])
def test_config_rejects_invalid_context_window(value: str) -> None:
    with pytest.raises(ConfigurationError, match='MODEL_CONTEXT_WINDOW'):
        ForgeConfig.from_env(
            {
                'ANTHROPIC_API_KEY': 'test-key',
                'MODEL_ID': 'test-model',
                'MODEL_MAX_TOKENS': '8192',
                'MODEL_CONTEXT_WINDOW': value,
            }
        )
