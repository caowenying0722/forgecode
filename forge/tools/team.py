'''Team messaging tools for lead and bounded subagents.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from forge.runtime.team import MessageBus, render_team_notification
from forge.tools.base import Tool, ToolExecutionError, ToolInput, ToolResult


class SendMessageInput(ToolInput):
    to: str = Field(min_length=1, max_length=80)
    type: Literal['status', 'question', 'result', 'warning']
    content: str = Field(min_length=1, max_length=8_000)


class SendMessageTool(Tool[SendMessageInput]):
    name = 'send_message'
    description = (
        'Send a durable team message to another agent inbox. Use from '
        'bounded subagents to report status, ask a question, return an '
        'intermediate result, or warn the lead. Common recipient: lead. '
        'Messages are delivered through .forge/teams and injected into the '
        'recipient context when collected.'
    )
    input_model = SendMessageInput

    def __init__(
        self,
        root: Path,
        *,
        bus: MessageBus | None = None,
        sender: str = 'lead',
    ) -> None:
        super().__init__(root)
        self.bus = bus or MessageBus(root)
        self.sender = sender

    async def execute(self, arguments: SendMessageInput) -> ToolResult:
        try:
            message = self.bus.send(
                sender=self.sender,
                recipient=arguments.to,
                message_type=arguments.type,
                content=arguments.content,
            )
        except ValueError as error:
            raise ToolExecutionError('team_message_rejected', str(error)) from error
        return ToolResult.ok(
            f'Sent {arguments.type} message to {arguments.to}.',
            content=json.dumps(message.as_dict(), ensure_ascii=False, indent=2),
            metadata={
                'message_id': message.id,
                'to': message.recipient,
                'from': message.sender,
                'type': message.type,
            },
        )


class CheckInboxInput(ToolInput):
    pass


class CheckInboxTool(Tool[CheckInboxInput]):
    name = 'check_inbox'
    description = (
        'Collect and consume durable team messages addressed to this agent. '
        'Use when a subagent needs lead replies or when the lead wants to '
        'manually inspect queued team messages. Lead messages are also '
        'automatically injected before each model request.'
    )
    input_model = CheckInboxInput

    def __init__(
        self,
        root: Path,
        *,
        bus: MessageBus | None = None,
        recipient: str = 'lead',
    ) -> None:
        super().__init__(root)
        self.bus = bus or MessageBus(root)
        self.recipient = recipient

    async def execute(self, arguments: CheckInboxInput) -> ToolResult:
        del arguments
        try:
            messages = self.bus.collect(self.recipient)
        except ValueError as error:
            raise ToolExecutionError('inbox_rejected', str(error)) from error
        if not messages:
            return ToolResult.ok(
                f'Inbox for {self.recipient} is empty.',
                metadata={'recipient': self.recipient, 'message_count': 0},
            )
        return ToolResult.ok(
            f'Collected {len(messages)} team message(s) for {self.recipient}.',
            content='\n'.join(render_team_notification(messages)),
            metadata={
                'recipient': self.recipient,
                'message_count': len(messages),
                'message_ids': [message.id for message in messages],
            },
        )


def create_team_tools(
    root: Path,
    *,
    bus: MessageBus | None = None,
    agent_id: str = 'lead',
) -> tuple[SendMessageTool, CheckInboxTool]:
    shared = bus or MessageBus(root)
    return (
        SendMessageTool(root, bus=shared, sender=agent_id),
        CheckInboxTool(root, bus=shared, recipient=agent_id),
    )
