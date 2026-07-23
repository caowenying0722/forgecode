'''Tests for ForgeCode hook registration and dispatch.'''

from __future__ import annotations

from pathlib import Path

from forge.hooks.registry import HookRegistry
from forge.hooks.state import HookContext, HookResult
from forge.runtime.state import ToolCall
from forge.tools.base import ToolResult


class BlockingHook:
    name = 'blocker'
    events = ('pre_tool_use',)
    description = 'Block test tool calls.'
    enabled = True

    async def handle(self, context: HookContext) -> HookResult:
        assert context.tool_call is not None
        return HookResult(
            tool_result=ToolResult.fail(
                'blocked_by_hook',
                f'Blocked {context.tool_call.name}.',
            )
        )


class RecordingHook:
    name = 'recorder'
    events = ('user_prompt_submit', 'stop')
    description = 'Record turn events.'
    enabled = True

    def __init__(self) -> None:
        self.seen_events: list[str] = []

    async def handle(self, context: HookContext) -> HookResult:
        self.seen_events.append(context.event)
        return HookResult()


def test_hook_registry_runs_registered_hooks_in_order(tmp_path: Path) -> None:
    registry = HookRegistry([BlockingHook()])

    result = run(
        registry.run(
            HookContext(
                event='pre_tool_use',
                root=tmp_path,
                tool_call=ToolCall(0, 'toolu_test', 'write_file', {}),
                effect='workspace_write',
            )
        )
    )

    assert result.tool_result is not None
    assert result.tool_result.error is not None
    assert result.tool_result.error.code == 'blocked_by_hook'


def test_hook_registry_describes_registered_hooks() -> None:
    registry = HookRegistry([BlockingHook()])

    description = registry.describe()

    assert 'Registered hooks:' in description
    assert 'blocker [enabled] on pre_tool_use' in description


def test_hook_registry_supports_turn_events(tmp_path: Path) -> None:
    hook = RecordingHook()
    registry = HookRegistry([hook])

    run(
        registry.run(
            HookContext(
                event='user_prompt_submit',
                root=tmp_path,
                prompt='hello',
            )
        )
    )
    run(registry.run(HookContext(event='stop', root=tmp_path)))

    assert hook.seen_events == ['user_prompt_submit', 'stop']


def run(coro):
    import asyncio

    return asyncio.run(coro)
