'''Minimal model execution for the M1 Agent Loop.'''

from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path

from anthropic.types import MessageParam

from forge.runtime.model_client import AnthropicModelClient, ModelClient
from forge.runtime.state import (
    ConversationEvent,
    ModelTextDelta,
    ModelUsageUpdate,
    TokenUsage,
    TurnCompleted,
    TurnResult,
)


class ModelResponseError(RuntimeError):
    '''Raised when a model response cannot be shown as text.'''


@cache
def load_system_prompt() -> str:
    '''Load the packaged ForgeCode identity and behavior prompt.'''
    prompt_path = Path(__file__).resolve().parents[1] / 'prompts' / 'system.md'
    prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not prompt:
        raise RuntimeError('ForgeCode system prompt is empty.')
    return prompt


class Conversation:
    '''Keep model-visible message history for an interactive session.'''

    def __init__(
        self,
        client: ModelClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = (
            client if client is not None else AnthropicModelClient.from_config()
        )
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else load_system_prompt()
        )
        self.messages: list[MessageParam] = []

    async def stream(self, prompt: str) -> AsyncIterator[ConversationEvent]:
        '''Stream one turn and commit it after a valid final result.'''
        if not prompt.strip():
            raise ValueError('prompt must not be empty')

        user_message: MessageParam = {'role': 'user', 'content': prompt}
        request_messages = [*self.messages, user_message]
        text_parts: list[str] = []
        final_usage: TokenUsage | None = None

        async for event in self.client.stream(
            messages=request_messages,
            system=self.system_prompt,
        ):
            if isinstance(event, ModelTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, ModelUsageUpdate):
                final_usage = event.usage
            yield event

        text = ''.join(text_parts).strip()
        if not text:
            raise ModelResponseError(
                'Model response did not contain any text.'
            )
        if final_usage is None:
            raise ModelResponseError(
                'Model response did not contain token usage.'
            )

        assistant_message: MessageParam = {
            'role': 'assistant',
            'content': text,
        }
        self.messages.extend([user_message, assistant_message])
        yield TurnCompleted(
            result=TurnResult(text=text, usage=final_usage)
        )
