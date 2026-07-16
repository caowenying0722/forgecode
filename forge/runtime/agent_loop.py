'''Minimal model execution for the M1 Agent Loop.'''

from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path

from anthropic.types import ContentBlockParam, MessageParam, ToolParam

from forge.runtime.model_client import AnthropicModelClient, ModelClient
from forge.runtime.state import (
    ConversationEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    TurnCompleted,
    TurnResult,
    ToolCall,
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
        tools: list[ToolParam] | None = None,
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
        self.tools = tools

    async def stream(self, prompt: str) -> AsyncIterator[ConversationEvent]:
        '''Stream one turn and commit it after a valid final result.'''
        if not prompt.strip():
            raise ValueError('prompt must not be empty')

        user_message: MessageParam = {'role': 'user', 'content': prompt}
        request_messages = [*self.messages, user_message]
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final_usage: TokenUsage | None = None

        async for event in self.client.stream(
            messages=request_messages,
            tools=self.tools,
            system=self.system_prompt,
        ):
            if isinstance(event, ModelTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, ModelToolCallCompleted):
                tool_calls.append(event.tool_call)
            elif isinstance(event, ModelUsageUpdate):
                final_usage = event.usage
            yield event

        text = ''.join(text_parts).strip()
        if not text and not tool_calls:
            raise ModelResponseError(
                'Model response did not contain any text or tool calls.'
            )
        if final_usage is None:
            raise ModelResponseError(
                'Model response did not contain token usage.'
            )

        assistant_message = build_assistant_message(text, tool_calls)
        self.messages.extend([user_message, assistant_message])
        yield TurnCompleted(
            result=TurnResult(
                text=text,
                usage=final_usage,
                tool_calls=tuple(tool_calls),
            )
        )


def build_assistant_message(
    text: str,
    tool_calls: list[ToolCall],
) -> MessageParam:
    '''Build model-visible assistant history from a completed response.'''
    if not tool_calls:
        return {'role': 'assistant', 'content': text}

    content: list[ContentBlockParam] = []
    if text:
        content.append({'type': 'text', 'text': text})
    content.extend(
        {
            'type': 'tool_use',
            'id': call.id,
            'name': call.name,
            'input': call.arguments,
        }
        for call in sorted(tool_calls, key=lambda call: call.index)
    )
    return {'role': 'assistant', 'content': content}
