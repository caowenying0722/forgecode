'''Tools for model-managed repository memory.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from forge.context.repository import MEMORY_TYPES, MemoryStore
from forge.tools.base import Tool, ToolExecutionError, ToolInput, ToolResult


MemoryType = Literal['user', 'feedback', 'project', 'reference']


class MemoryListInput(ToolInput):
    pass


class MemoryReadInput(ToolInput):
    name: str = Field(min_length=1)


class MemoryWriteInput(ToolInput):
    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=20_000)
    description: str = Field(default='', max_length=500)
    memory_type: MemoryType = 'project'


class MemoryUpdateInput(ToolInput):
    name: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=20_000)
    description: str = Field(default='', max_length=500)
    memory_type: MemoryType | None = None


class MemoryDeleteInput(ToolInput):
    name: str = Field(min_length=1)


class MemoryListTool(Tool[MemoryListInput]):
    name = 'memory_list'
    description = (
        'List durable repository memory records available to ForgeCode. Use '
        'before reading or updating memory when the exact name is unknown.'
    )
    input_model = MemoryListInput

    def __init__(self, root: Path, store: MemoryStore | None = None) -> None:
        super().__init__(root)
        self.store = store or MemoryStore(self.root)

    async def execute(self, arguments: MemoryListInput) -> ToolResult:
        del arguments
        records = self.store.list()
        content = json.dumps(
            [
                {
                    'name': record.name,
                    'description': record.description,
                    'type': record.memory_type,
                    'source': record.source,
                    'created_at': record.created_at,
                    'updated_at': record.updated_at,
                    'path': record.path.relative_to(self.root).as_posix(),
                }
                for record in records
            ],
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult.ok(
            f'Listed {len(records)} memory record(s).',
            content=content,
            metadata={'memory_count': len(records)},
        )


class MemoryReadTool(Tool[MemoryReadInput]):
    name = 'memory_read'
    description = 'Read one durable repository memory record by name.'
    input_model = MemoryReadInput

    def __init__(self, root: Path, store: MemoryStore | None = None) -> None:
        super().__init__(root)
        self.store = store or MemoryStore(self.root)

    async def execute(self, arguments: MemoryReadInput) -> ToolResult:
        record = self.store.get(arguments.name)
        if record is None:
            raise ToolExecutionError(
                'memory_not_found',
                f'Memory not found: {arguments.name}',
            )
        content = json.dumps(
            {
                'name': record.name,
                'description': record.description,
                'type': record.memory_type,
                'source': record.source,
                'created_at': record.created_at,
                'updated_at': record.updated_at,
                'path': record.path.relative_to(self.root).as_posix(),
                'content': record.content,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult.ok(
            f'Read memory {record.name}.',
            content=content,
            metadata={'memory_name': record.name},
        )


class MemoryWriteTool(Tool[MemoryWriteInput]):
    name = 'memory_write'
    description = (
        'Create a durable repository memory record. Use only for stable facts, '
        'project conventions, or explicit user preferences that should survive '
        'future sessions. Do not store secrets.'
    )
    input_model = MemoryWriteInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: MemoryStore | None = None) -> None:
        super().__init__(root)
        self.store = store or MemoryStore(self.root)

    async def execute(self, arguments: MemoryWriteInput) -> ToolResult:
        try:
            record = self.store.create(
                arguments.name,
                arguments.content,
                description=arguments.description,
                memory_type=arguments.memory_type,
                source='model_memory_tool',
            )
        except ValueError as error:
            raise ToolExecutionError('memory_write_rejected', str(error)) from error
        return ToolResult.ok(
            f'Created memory {record.name}.',
            content=record.path.relative_to(self.root).as_posix(),
            metadata={'memory_name': record.name, 'memory_write': True},
        )


class MemoryUpdateTool(Tool[MemoryUpdateInput]):
    name = 'memory_update'
    description = (
        'Update an existing durable repository memory record. Read or list '
        'memory first if the current name or contents are unclear.'
    )
    input_model = MemoryUpdateInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: MemoryStore | None = None) -> None:
        super().__init__(root)
        self.store = store or MemoryStore(self.root)

    async def execute(self, arguments: MemoryUpdateInput) -> ToolResult:
        if (
            arguments.memory_type is not None
            and arguments.memory_type not in MEMORY_TYPES
        ):
            raise ToolExecutionError(
                'memory_update_rejected',
                f'Unsupported memory type: {arguments.memory_type}',
            )
        try:
            record = self.store.update(
                arguments.name,
                arguments.content,
                description=arguments.description,
                memory_type=arguments.memory_type,
                source='model_memory_tool',
            )
        except ValueError as error:
            raise ToolExecutionError('memory_update_rejected', str(error)) from error
        return ToolResult.ok(
            f'Updated memory {record.name}.',
            content=record.path.relative_to(self.root).as_posix(),
            metadata={'memory_name': record.name, 'memory_write': True},
        )


class MemoryDeleteTool(Tool[MemoryDeleteInput]):
    name = 'memory_delete'
    description = 'Delete one durable repository memory record by name.'
    input_model = MemoryDeleteInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: MemoryStore | None = None) -> None:
        super().__init__(root)
        self.store = store or MemoryStore(self.root)

    async def execute(self, arguments: MemoryDeleteInput) -> ToolResult:
        if not self.store.forget(arguments.name):
            raise ToolExecutionError(
                'memory_not_found',
                f'Memory not found: {arguments.name}',
            )
        return ToolResult.ok(
            f'Deleted memory {arguments.name}.',
            metadata={'memory_name': arguments.name, 'memory_write': True},
        )


def create_memory_tools(root: Path) -> tuple[
    MemoryListTool,
    MemoryReadTool,
    MemoryWriteTool,
    MemoryUpdateTool,
    MemoryDeleteTool,
]:
    store = MemoryStore(root)
    return (
        MemoryListTool(root, store),
        MemoryReadTool(root, store),
        MemoryWriteTool(root, store),
        MemoryUpdateTool(root, store),
        MemoryDeleteTool(root, store),
    )
