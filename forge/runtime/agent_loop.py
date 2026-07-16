'''Multi-step model and tool execution for the M1 Agent Loop.'''

from collections.abc import AsyncIterator
from functools import cache
import json
from pathlib import Path

from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    ToolParam,
    ToolResultBlockParam,
)

from forge.runtime.model_client import AnthropicModelClient, ModelClient
from forge.runtime.state import (
    ConversationEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnCompleted,
    TurnResult,
    ToolCall,
)
from forge.tools.base import ToolRegistry, ToolResult


class ModelResponseError(RuntimeError):
    '''Raised when a model response cannot continue the Agent Loop.'''


class AgentLoopLimitError(RuntimeError):
    '''Raised when one user turn exceeds its model-call safety limit.'''


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
        registry: ToolRegistry | None = None,
        max_iterations: int = 20,
    ) -> None:
        if tools is not None and registry is not None:
            raise ValueError('Pass tools or registry, not both.')
        if max_iterations < 1:
            raise ValueError('max_iterations must be positive')
        self.client = (
            client if client is not None else AnthropicModelClient.from_config()
        )
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else load_system_prompt()
        )
        self.messages: list[MessageParam] = []
        self.registry = registry
        self.tools = registry.definitions if registry is not None else tools
        self.max_iterations = max_iterations

    async def stream(self, prompt: str) -> AsyncIterator[ConversationEvent]:
        '''Run model-tool cycles until the model returns a final text answer.'''
        if not prompt.strip():
            raise ValueError('prompt must not be empty')

        user_message: MessageParam = {'role': 'user', 'content': prompt}
        request_messages = [*self.messages, user_message]
        completed_usage = TokenUsage(input_tokens=0, output_tokens=0)
        all_tool_calls: list[ToolCall] = []

        for _ in range(self.max_iterations):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            request_usage: TokenUsage | None = None

            async for event in self.client.stream(
                messages=[*request_messages],
                tools=self.tools,
                system=self.system_prompt,
            ):
                if isinstance(event, ModelTextDelta):
                    text_parts.append(event.text)
                    yield event
                elif isinstance(event, ModelToolCallCompleted):
                    tool_calls.append(event.tool_call)
                    yield event
                elif isinstance(event, ModelUsageUpdate):
                    request_usage = event.usage
                    yield ModelUsageUpdate(
                        usage=add_token_usage(completed_usage, request_usage)
                    )
                else:
                    yield event

            text = ''.join(text_parts).strip()
            if not text and not tool_calls:
                raise ModelResponseError(
                    'Model response did not contain any text or tool calls.'
                )
            if request_usage is None:
                raise ModelResponseError(
                    'Model response did not contain token usage.'
                )

            completed_usage = add_token_usage(
                completed_usage,
                request_usage,
            )
            tool_calls.sort(key=lambda call: call.index)
            request_messages.append(
                build_assistant_message(text, tool_calls)
            )

            if not tool_calls:
                self.messages = request_messages
                yield TurnCompleted(
                    result=TurnResult(
                        text=text,
                        usage=completed_usage,
                        tool_calls=tuple(all_tool_calls),
                    )
                )
                return

            if self.registry is None:
                raise ModelResponseError(
                    'Model requested tools, but no ToolRegistry is configured.'
                )

            all_tool_calls.extend(tool_calls)
            tool_results: list[tuple[ToolCall, ToolResult]] = []
            for tool_call in tool_calls:
                yield ToolExecutionStarted(tool_call=tool_call)
                result = await self.registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                )
                tool_results.append((tool_call, result))
                yield ToolExecutionCompleted(
                    tool_call=tool_call,
                    result=result,
                )
            request_messages.append(build_tool_result_message(tool_results))

        raise AgentLoopLimitError(
            f'Agent Loop exceeded {self.max_iterations} model calls.'
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


def build_tool_result_message(
    tool_results: list[tuple[ToolCall, ToolResult]],
) -> MessageParam:
    '''Build one user message containing ordered Anthropic tool results.'''
    content: list[ToolResultBlockParam] = []
    for tool_call, result in tool_results:
        content.append(
            {
                'type': 'tool_result',
                'tool_use_id': tool_call.id,
                'content': serialize_tool_result(result),
                'is_error': not result.success,
            }
        )
    return {'role': 'user', 'content': content}


def serialize_tool_result(result: ToolResult) -> str:
    '''Serialize the stable ToolResult contract for model consumption.'''
    error = None
    if result.error is not None:
        error = {
            'code': result.error.code,
            'message': result.error.message,
            'details': result.error.details,
        }
    return json.dumps(
        {
            'success': result.success,
            'summary': result.summary,
            'content': result.content,
            'error': error,
            'metadata': result.metadata,
        },
        ensure_ascii=False,
        default=str,
    )


def add_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    '''Add exact usage from separate model requests in one user turn.'''
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_creation_input_tokens=(
            left.cache_creation_input_tokens
            + right.cache_creation_input_tokens
        ),
        cache_read_input_tokens=(
            left.cache_read_input_tokens + right.cache_read_input_tokens
        ),
    )
