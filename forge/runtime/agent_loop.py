'''Minimal model execution for the M1 Agent Loop.'''

from functools import cache
from pathlib import Path

from anthropic.types import Message, MessageParam

from forge.runtime.model_client import AnthropicModelClient, ModelClient


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

    async def send(self, prompt: str) -> str:
        '''Send one turn and commit it to history after a valid response.'''
        if not prompt.strip():
            raise ValueError('prompt must not be empty')

        user_message: MessageParam = {'role': 'user', 'content': prompt}
        request_messages = [*self.messages, user_message]
        response = await self.client.generate(
            messages=request_messages,
            system=self.system_prompt,
        )
        text = extract_text(response)
        assistant_message: MessageParam = {
            'role': 'assistant',
            'content': text,
        }
        self.messages.extend([user_message, assistant_message])
        return text


def extract_text(message: Message) -> str:
    '''Extract displayable text blocks from an Anthropic message.'''
    text = '\n'.join(
        block.text for block in message.content if block.type == 'text'
    ).strip()
    if not text:
        raise ModelResponseError('Model response did not contain any text.')
    return text


async def run_single_turn(
    prompt: str,
    client: ModelClient | None = None,
) -> str:
    '''Send one user prompt and return the model's text response.'''
    if not prompt.strip():
        raise ValueError('prompt must not be empty')
    return await Conversation(client=client).send(prompt)
