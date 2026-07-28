'''Team messaging tools for lead and bounded subagents.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from forge.runtime.team import (
    MessageBus,
    TeamRequest,
    render_team_notification,
)
from forge.tasks.graph import TaskGraphStore
from forge.tools.task_graph import render_task_json
from forge.tools.base import Tool, ToolExecutionError, ToolInput, ToolResult


class SendMessageInput(ToolInput):
    to: str = Field(min_length=1, max_length=80)
    type: Literal['status', 'question', 'result', 'warning']
    content: str = Field(min_length=1, max_length=8_000)
    reply_to: str | None = Field(default=None)


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
                reply_to=arguments.reply_to,
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


class RequestTeamActionInput(ToolInput):
    to: str = Field(min_length=1, max_length=80)
    type: Literal['shutdown', 'plan_approval', 'task_assignment', 'custom']
    content: str = Field(min_length=1, max_length=8_000)


class RequestTeamActionTool(Tool[RequestTeamActionInput]):
    name = 'request_team_action'
    description = (
        'Start a structured team request/response protocol. Creates a '
        'durable request_id, sends a typed request message, and tracks '
        'pending/approved/rejected state under .forge/teams. Use for '
        'shutdown handshakes, plan approval, task assignment negotiation, '
        'or other coordination that must be correlated with a response.'
    )
    input_model = RequestTeamActionInput

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

    async def execute(self, arguments: RequestTeamActionInput) -> ToolResult:
        try:
            request, message = self.bus.request(
                sender=self.sender,
                recipient=arguments.to,
                request_type=arguments.type,
                content=arguments.content,
            )
        except ValueError as error:
            raise ToolExecutionError('team_request_rejected', str(error)) from error
        return ToolResult.ok(
            f'Sent {arguments.type} request to {arguments.to}.',
            content=json.dumps(request.as_dict(), ensure_ascii=False, indent=2),
            metadata={
                'request_id': request.request_id,
                'message_id': message.id,
                'to': request.target,
                'from': request.sender,
                'type': request.type,
                'status': request.status,
            },
        )


class RespondTeamRequestInput(ToolInput):
    request_id: str = Field(min_length=1)
    approve: bool
    content: str = Field(min_length=1, max_length=8_000)
    reply_to: str | None = Field(default=None)


class RespondTeamRequestTool(Tool[RespondTeamRequestInput]):
    name = 'respond_team_request'
    description = (
        'Approve or reject a pending team protocol request addressed to this '
        'agent. The response is correlated by request_id and updates durable '
        'request state before sending the response message.'
    )
    input_model = RespondTeamRequestInput

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

    async def execute(self, arguments: RespondTeamRequestInput) -> ToolResult:
        try:
            request, message = self.bus.respond(
                sender=self.sender,
                request_id=arguments.request_id,
                approve=arguments.approve,
                content=arguments.content,
                reply_to=arguments.reply_to,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ToolExecutionError('team_response_rejected', str(error)) from error
        return ToolResult.ok(
            f'Request {request.request_id} {request.status}.',
            content=json.dumps(message.as_dict(), ensure_ascii=False, indent=2),
            metadata={
                'request_id': request.request_id,
                'message_id': message.id,
                'status': request.status,
                'to': message.recipient,
                'from': message.sender,
            },
        )


class ListTeamRequestsInput(ToolInput):
    status: Literal['pending', 'approved', 'rejected'] | None = None


class ListTeamRequestsTool(Tool[ListTeamRequestsInput]):
    name = 'list_team_requests'
    description = (
        'List durable team protocol requests and their pending/approved/'
        'rejected states. Use to recover coordination state or inspect '
        'outstanding shutdown, plan approval, or task assignment requests.'
    )
    input_model = ListTeamRequestsInput

    def __init__(
        self,
        root: Path,
        *,
        bus: MessageBus | None = None,
    ) -> None:
        super().__init__(root)
        self.bus = bus or MessageBus(root)

    async def execute(self, arguments: ListTeamRequestsInput) -> ToolResult:
        requests = self.bus.list_requests(status=arguments.status)
        if not requests:
            return ToolResult.ok(
                'No team protocol requests found.',
                metadata={'request_count': 0},
            )
        return ToolResult.ok(
            f'Listed {len(requests)} team request(s).',
            content='\n'.join(render_request_line(request) for request in requests),
            metadata={
                'request_count': len(requests),
                'request_ids': [request.request_id for request in requests],
            },
        )


class ClaimNextTaskInput(ToolInput):
    owner: str | None = Field(default=None, max_length=200)


class ClaimNextTaskTool(Tool[ClaimNextTaskInput]):
    name = 'claim_next_task'
    description = (
        'Scan the durable task graph and claim the first pending task whose '
        'dependencies are completed. Use from autonomous/bounded teammates '
        'when an idle teammate should self-claim and self-organize from the task board. '
        'This does not create tasks and will report when no task is available.'
    )
    input_model = ClaimNextTaskInput
    effect = 'workspace_write'

    def __init__(
        self,
        root: Path,
        *,
        store: TaskGraphStore | None = None,
        agent_id: str = 'lead',
    ) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)
        self.agent_id = agent_id

    async def execute(self, arguments: ClaimNextTaskInput) -> ToolResult:
        owner = arguments.owner or self.agent_id
        for task in self.store.ready_wave():
            try:
                claimed = self.store.claim(task.id, owner=owner)
            except ValueError:
                continue
            return ToolResult.ok(
                f'Claimed next task {claimed.id}.',
                content=render_task_json(claimed),
                metadata={'task_id': claimed.id, 'owner': claimed.owner},
            )
        return ToolResult.ok(
            'No claimable task-graph item is available.',
            metadata={'task_id': None, 'owner': owner},
        )


def create_team_tools(
    root: Path,
    *,
    bus: MessageBus | None = None,
    agent_id: str = 'lead',
) -> tuple[
    SendMessageTool,
    CheckInboxTool,
    RequestTeamActionTool,
    RespondTeamRequestTool,
    ListTeamRequestsTool,
    ClaimNextTaskTool,
]:
    shared = bus or MessageBus(root)
    return (
        SendMessageTool(root, bus=shared, sender=agent_id),
        CheckInboxTool(root, bus=shared, recipient=agent_id),
        RequestTeamActionTool(root, bus=shared, sender=agent_id),
        RespondTeamRequestTool(root, bus=shared, sender=agent_id),
        ListTeamRequestsTool(root, bus=shared),
        ClaimNextTaskTool(root, agent_id=agent_id),
    )


def render_request_line(request: TeamRequest) -> str:
    return (
        f'- {request.request_id} [{request.status}] {request.type}: '
        f'{request.sender} -> {request.target}'
    )
