'''Hook registration and built-in runtime hooks for ForgeCode.'''

from forge.hooks.builtin import PermissionHook, TodoPlanningHook, ToolLoggingHook
from forge.hooks.registry import HookRegistry
from forge.hooks.state import (
    HookContext,
    HookEvent,
    HookResult,
    RegisteredHook,
)

__all__ = [
    'HookContext',
    'HookEvent',
    'HookRegistry',
    'HookResult',
    'PermissionHook',
    'RegisteredHook',
    'TodoPlanningHook',
    'ToolLoggingHook',
]
