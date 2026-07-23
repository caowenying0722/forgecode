'''Repository instructions and durable Markdown memory.'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


MEMORY_TYPES = frozenset({'user', 'feedback', 'project', 'reference'})
SECRET_PATTERN = re.compile(
    r'(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+'
    r'|sk-[a-z0-9_-]{8,}|-----BEGIN [A-Z ]+PRIVATE KEY-----'
)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    name: str
    description: str
    memory_type: str
    source: str
    created_at: str
    updated_at: str
    content: str
    path: Path


@dataclass(frozen=True, slots=True)
class MemorySelection:
    records: tuple[MemoryRecord, ...]
    rendered: str


class MemoryStore:
    '''Store durable project knowledge as inspectable Markdown files.'''

    def __init__(
        self,
        root: Path,
        *,
        max_selected: int = 5,
        max_file_characters: int = 4_096,
        max_total_characters: int = 20_000,
    ) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'memory'
        self.index_path = self.directory / 'MEMORY.md'
        self.max_selected = max_selected
        self.max_file_characters = max_file_characters
        self.max_total_characters = max_total_characters

    def list(self) -> tuple[MemoryRecord, ...]:
        if not self.directory.exists():
            return ()
        records: list[MemoryRecord] = []
        for path in sorted(self.directory.glob('*.md')):
            if path.name == 'MEMORY.md':
                continue
            try:
                records.append(parse_memory(path))
            except ValueError:
                continue
        return tuple(records)

    def remember(
        self,
        name: str,
        content: str,
        *,
        description: str = '',
        memory_type: str = 'project',
        source: str = 'manual',
    ) -> MemoryRecord:
        clean_name = name.strip()
        clean_content = content.strip()
        if not clean_name:
            raise ValueError('Memory name must not be empty.')
        if not clean_content:
            raise ValueError('Memory content must not be empty.')
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f'Unsupported memory type: {memory_type}')
        if SECRET_PATTERN.search(clean_content):
            raise ValueError('Memory content appears to contain a secret.')
        existing = next(
            (record for record in self.list() if record.name == clean_name),
            None,
        )
        timestamp = utc_timestamp()
        created_at = timestamp if existing is None else existing.created_at
        clean_source = source.strip() or (
            'manual' if existing is None else existing.source
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = existing.path if existing is not None else (
            self.directory / f'{memory_slug(clean_name)}.md'
        )
        resolved_description = description.strip() or clean_content[:120]
        path.write_text(
            render_memory(
                clean_name,
                resolved_description,
                memory_type,
                clean_source,
                created_at,
                timestamp,
                clean_content,
            ),
            encoding='utf-8',
        )
        self.rebuild_index()
        if len(self.list()) >= 10:
            self.consolidate()
        return parse_memory(path)

    def get(self, name: str) -> MemoryRecord | None:
        normalized = name.strip().casefold()
        return next(
            (
                record
                for record in self.list()
                if record.name.casefold() == normalized
                or record.path.stem.casefold() == normalized
            ),
            None,
        )

    def forget(self, name: str) -> bool:
        record = self.get(name)
        if record is None:
            return False
        record.path.unlink()
        self.rebuild_index()
        return True

    def create(
        self,
        name: str,
        content: str,
        *,
        description: str = '',
        memory_type: str = 'project',
        source: str = 'manual',
    ) -> MemoryRecord:
        if self.get(name) is not None:
            raise ValueError(f'Memory already exists: {name}')
        return self.remember(
            name,
            content,
            description=description,
            memory_type=memory_type,
            source=source,
        )

    def update(
        self,
        name: str,
        content: str,
        *,
        description: str = '',
        memory_type: str | None = None,
        source: str = '',
    ) -> MemoryRecord:
        existing = self.get(name)
        if existing is None:
            raise ValueError(f'Memory not found: {name}')
        return self.remember(
            existing.name,
            content,
            description=description or existing.description,
            memory_type=memory_type or existing.memory_type,
            source=source or existing.source,
        )

    def select(self, query: str) -> MemorySelection:
        query_terms = search_terms(query)
        if not query_terms:
            return MemorySelection((), '')
        ranked: list[tuple[int, str, MemoryRecord]] = []
        for record in self.list():
            score = (
                5 * len(query_terms & search_terms(record.name))
                + 3 * len(query_terms & search_terms(record.description))
                + len(query_terms & search_terms(record.content))
            )
            if score:
                ranked.append((score, record.name.casefold(), record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected: list[MemoryRecord] = []
        sections: list[str] = []
        total = 0
        for _, _, record in ranked[:self.max_selected]:
            content = record.content[:self.max_file_characters]
            section = f'### {record.name}\n{content}'
            if total + len(section) > self.max_total_characters:
                break
            selected.append(record)
            sections.append(section)
            total += len(section)
        rendered = '\n\n'.join(sections)
        return MemorySelection(tuple(selected), rendered)

    def rebuild_index(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        lines = ['# ForgeCode Repository Memory', '']
        for record in self.list():
            relative = record.path.name
            lines.append(
                f'- [{record.name}]({relative}) — '
                f'{record.description} ({record.memory_type})'
            )
        self.index_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return self.index_path

    def consolidate(self) -> int:
        '''Remove exact duplicate content and rebuild the stable index.'''
        seen: dict[str, MemoryRecord] = {}
        removed = 0
        for record in self.list():
            digest = hashlib.sha256(
                record.content.strip().encode('utf-8')
            ).hexdigest()
            if digest in seen:
                record.path.unlink()
                removed += 1
            else:
                seen[digest] = record
        self.rebuild_index()
        return removed


class RepositoryContext:
    '''Load scoped repository instructions and query-relevant memory.'''

    def __init__(
        self,
        root: Path,
        *,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.cwd = resolve_context_cwd(self.root, cwd)
        self.home = (home or Path.home()).resolve()
        self.memory = MemoryStore(self.root)

    def instructions(self) -> str:
        paths = list(instruction_paths(self.root, self.cwd, self.home))
        paths.append(self.root / 'FORGE.md')
        rules = self.root / '.forge' / 'rules'
        if rules.exists():
            paths.extend(sorted(rules.glob('*.md')))
        sections: list[str] = []
        total = 0
        for path in paths:
            if not path.is_file():
                continue
            content = path.read_text(encoding='utf-8')[:16_000]
            if total + len(content) > 32_000:
                break
            sections.append(
                f'## {display_instruction_path(path, self.root, self.home)}\n'
                f'{content}'
            )
            total += len(content)
        return '\n\n'.join(sections)

    def system_suffix(self, query: str) -> str:
        sections: list[str] = []
        instructions = self.instructions()
        if instructions:
            sections.append('# Repository instructions\n' + instructions)
        selected = self.memory.select(query)
        if selected.rendered:
            sections.append('# Relevant repository memory\n' + selected.rendered)
        return '\n\n'.join(sections)


def render_memory(
    name: str,
    description: str,
    memory_type: str,
    source: str,
    created_at: str,
    updated_at: str,
    content: str,
) -> str:
    return (
        '---\n'
        f'name: {json.dumps(name, ensure_ascii=False)}\n'
        f'description: {json.dumps(description, ensure_ascii=False)}\n'
        f'type: {json.dumps(memory_type)}\n'
        f'source: {json.dumps(source, ensure_ascii=False)}\n'
        f'created_at: {json.dumps(created_at)}\n'
        f'updated_at: {json.dumps(updated_at)}\n'
        '---\n\n'
        f'{content.strip()}\n'
    )


def parse_memory(path: Path) -> MemoryRecord:
    text = path.read_text(encoding='utf-8')
    if not text.startswith('---\n') or '\n---\n' not in text[4:]:
        raise ValueError(f'Invalid memory frontmatter: {path}')
    header, content = text[4:].split('\n---\n', 1)
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(':')
        if not separator:
            continue
        try:
            metadata[key.strip()] = str(json.loads(value.strip()))
        except json.JSONDecodeError:
            metadata[key.strip()] = value.strip()
    name = metadata.get('name', '').strip()
    memory_type = metadata.get('type', '').strip()
    if not name or memory_type not in MEMORY_TYPES:
        raise ValueError(f'Invalid memory metadata: {path}')
    now = utc_timestamp()
    return MemoryRecord(
        name=name,
        description=metadata.get('description', '').strip(),
        memory_type=memory_type,
        source=metadata.get('source', 'legacy').strip() or 'legacy',
        created_at=metadata.get('created_at', now).strip() or now,
        updated_at=metadata.get('updated_at', now).strip() or now,
        content=content.strip(),
        path=path,
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def memory_slug(name: str) -> str:
    ascii_parts = re.findall(r'[a-z0-9]+', name.casefold())
    if ascii_parts:
        return '-'.join(ascii_parts)[:64]
    digest = hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]
    return f'memory-{digest}'


def resolve_context_cwd(root: Path, cwd: Path | None) -> Path:
    candidate = (cwd or Path.cwd()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return root
    return candidate


def instruction_paths(root: Path, cwd: Path, home: Path) -> tuple[Path, ...]:
    paths: list[Path] = [home / '.forge' / 'AGENTS.md']
    directories = path_chain(root, cwd)
    for directory in directories:
        paths.append(directory / 'AGENTS.md')
        paths.append(directory / 'AGENTS.override.md')
    return tuple(dict.fromkeys(paths))


def path_chain(root: Path, cwd: Path) -> tuple[Path, ...]:
    try:
        relative = cwd.relative_to(root)
    except ValueError:
        return (root,)
    directories = [root]
    current = root
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return tuple(directories)


def display_instruction_path(path: Path, root: Path, home: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return '~/' + path.relative_to(home).as_posix()
    except ValueError:
        return path.as_posix()


def search_terms(text: str) -> set[str]:
    lowered = text.casefold()
    terms = set(re.findall(r'[a-z0-9_]{2,}', lowered))
    for chunk in re.findall(r'[\u4e00-\u9fff]+', lowered):
        if len(chunk) == 1:
            terms.add(chunk)
        else:
            terms.update(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return terms
