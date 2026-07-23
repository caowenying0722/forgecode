'''Ordered hook registry used by the runtime execution boundary.'''

from __future__ import annotations

from typing import Protocol

from forge.hooks.state import HookContext, HookEvent, HookResult, RegisteredHook


class Hook(Protocol):
    name: str
    events: tuple[HookEvent, ...]
    description: str
    enabled: bool

    async def handle(self, context: HookContext) -> HookResult:
        ...


class HookRegistry:
    '''Run registered hooks in deterministic order.'''

    def __init__(self, hooks: tuple[Hook, ...] | list[Hook] = ()) -> None:
        self._hooks: list[Hook] = []
        for hook in hooks:
            self.register(hook)

    def register(self, hook: Hook) -> None:
        if any(existing.name == hook.name for existing in self._hooks):
            raise ValueError(f'Duplicate hook name: {hook.name}')
        self._hooks.append(hook)

    async def run(self, context: HookContext) -> HookResult:
        current_tool_call = context.tool_call
        merged_metadata: dict[str, object] = {}
        for hook in self._hooks:
            if not hook.enabled or context.event not in hook.events:
                continue
            scoped = context
            if current_tool_call is not None:
                scoped = scoped.for_event(
                    context.event,
                    tool_call=current_tool_call,
                )
            result = await hook.handle(scoped)
            if result.metadata:
                merged_metadata.update(result.metadata)
            if result.tool_call is not None:
                current_tool_call = result.tool_call
            if result.tool_result is not None or result.stop:
                return HookResult(
                    tool_call=current_tool_call,
                    tool_result=result.tool_result,
                    stop=result.stop,
                    metadata={**merged_metadata, **(result.metadata or {})},
                )
        return HookResult(
            tool_call=current_tool_call,
            metadata=merged_metadata or None,
        )

    def describe(self) -> str:
        if not self._hooks:
            return 'No hooks registered.'
        lines = ['Registered hooks:']
        for hook in self._hooks:
            state = 'enabled' if hook.enabled else 'disabled'
            events = ', '.join(hook.events)
            description = f' - {hook.description}' if hook.description else ''
            lines.append(f'- {hook.name} [{state}] on {events}{description}')
        return '\n'.join(lines)

    @property
    def registered(self) -> tuple[RegisteredHook, ...]:
        return tuple(
            RegisteredHook(
                name=hook.name,
                events=hook.events,
                description=hook.description,
                enabled=hook.enabled,
            )
            for hook in self._hooks
        )
