'''Tests for the ForgeCode CLI.'''

from pathlib import Path

import pytest
from typer.testing import CliRunner

import forge.cli as cli_module
from forge.cli import app
from forge.config import ConfigurationError


runner = CliRunner()


class FakeConversation:
    '''Return scripted responses for interactive CLI tests.'''

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def send(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_cli_starts_an_interactive_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FakeConversation('Hello', 'I remember')
    monkeypatch.setattr(cli_module, 'Conversation', lambda: conversation)

    result = runner.invoke(app, input='first\nsecond\n')

    assert result.exit_code == 0
    assert 'ForgeCode v0.1.0' in result.output
    assert 'Ctrl+C to exit' in result.output
    assert 'Ask a question or describe a coding task.' in result.output
    assert 'Hello' in result.output
    assert 'I remember' in result.output
    assert 'Session ended.' in result.output
    assert conversation.prompts == ['first', 'second']


def test_interactive_conversation_continues_after_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FakeConversation(
        RuntimeError('provider unavailable'),
        'Recovered',
    )
    monkeypatch.setattr(cli_module, 'Conversation', lambda: conversation)

    result = runner.invoke(app, input='first\nsecond\n')

    assert result.exit_code == 0
    assert 'Model request failed: provider unavailable' in result.output
    assert 'Recovered' in result.output
    assert conversation.prompts == ['first', 'second']


def test_interactive_conversation_explains_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_conversation() -> None:
        raise ConfigurationError('ANTHROPIC_API_KEY is not set.')

    monkeypatch.setattr(cli_module, 'Conversation', missing_conversation)

    result = runner.invoke(app)

    assert result.exit_code == 1
    assert 'Model configuration is incomplete.' in result.output
    assert 'ANTHROPIC_API_KEY is not set.' in result.output


def test_cli_help() -> None:
    result = runner.invoke(app, ['--help'])

    assert result.exit_code == 0
    assert 'ForgeCode terminal Agent Harness.' in result.stdout
    assert '--version' in result.stdout
    assert '--prompt' in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ['--version'])

    assert result.exit_code == 0
    assert 'ForgeCode 0.1.0' in result.stdout


def test_prompt_option_prints_single_turn_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    async def fake_run_single_turn(prompt: str) -> str:
        prompts.append(prompt)
        return 'READY'

    monkeypatch.setattr(cli_module, 'run_single_turn', fake_run_single_turn)

    result = runner.invoke(app, ['-p', 'Only reply READY'])

    assert result.exit_code == 0
    assert result.stdout == 'READY\n'
    assert prompts == ['Only reply READY']


def test_prompt_option_rejects_empty_prompt() -> None:
    result = runner.invoke(app, ['--prompt', '   '])

    assert result.exit_code == 2
    assert 'Prompt must not be empty.' in result.output


def test_prompt_option_reports_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_run_single_turn(prompt: str) -> str:
        raise RuntimeError('provider unavailable')

    monkeypatch.setattr(cli_module, 'run_single_turn', failing_run_single_turn)

    result = runner.invoke(app, ['-p', 'hello'])

    assert result.exit_code == 1
    assert 'Model request failed: provider unavailable' in result.output


def test_prompt_option_explains_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_config(prompt: str) -> str:
        raise ConfigurationError('ANTHROPIC_API_KEY is not set.')

    monkeypatch.setattr(cli_module, 'run_single_turn', missing_config)

    result = runner.invoke(app, ['-p', 'hello'])

    assert result.exit_code == 1
    assert 'Model configuration is incomplete.' in result.output
    assert 'ANTHROPIC_API_KEY is not set.' in result.output


def test_config_command_reports_ready_without_exposing_api_key() -> None:
    result = runner.invoke(
        app,
        ['config'],
        env={
            'ANTHROPIC_API_KEY': 'secret-test-key',
            'MODEL_ID': 'claude-test',
            'ANTHROPIC_BASE_URL': 'https://gateway.example.com/anthropic/',
        },
    )

    assert result.exit_code == 0
    assert 'Anthropic configuration is ready.' in result.stdout
    assert 'Model ID: claude-test' in result.stdout
    assert 'https://gateway.example.com/anthropic' in result.stdout
    assert 'API key: configured' in result.stdout
    assert 'secret-test-key' not in result.stdout


def test_config_command_explains_missing_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('MODEL_ID', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)

    result = runner.invoke(app, ['config'])

    assert result.exit_code == 1
    assert 'Model configuration is incomplete.' in result.output
    assert 'ANTHROPIC_API_KEY is not set.' in result.output


def test_config_command_explains_missing_model_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.delenv('MODEL_ID', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)

    result = runner.invoke(app, ['config'])

    assert result.exit_code == 1
    assert 'MODEL_ID is not set.' in result.output
