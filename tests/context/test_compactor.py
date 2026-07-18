'''Tests for cheap context compaction.'''

from pathlib import Path

from forge.context.compactor import CompactionConfig, cheap_compact


def tool_pair(call_id: str, result: str) -> list[dict[str, object]]:
    return [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': call_id,
                    'name': 'read_file',
                    'input': {'path': 'a.py'},
                }
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': call_id,
                    'content': result,
                }
            ],
        },
    ]


def test_large_tool_result_is_persisted_without_mutating_source(
    tmp_path: Path,
) -> None:
    messages = tool_pair('toolu_big', 'x' * 40)
    config = CompactionConfig(tool_result_inline_limit=20)

    result = cheap_compact(messages, tmp_path / 'artifacts', config)

    compacted = result.messages[1]['content'][0]['content']
    assert compacted.startswith('[ForgeCode stored a large tool result]')
    assert result.artifacts
    assert Path(result.artifacts[0]).read_text(encoding='utf-8') == 'x' * 40
    assert messages[1]['content'][0]['content'] == 'x' * 40


def test_old_tool_results_are_shortened_but_recent_results_stay() -> None:
    messages: list[dict[str, object]] = []
    for index in range(5):
        messages.extend(tool_pair(f'toolu_{index}', str(index) * 30))
    config = CompactionConfig(
        tool_result_inline_limit=1_000,
        old_tool_result_limit=10,
        keep_recent_tool_results=2,
    )

    result = cheap_compact(messages, Path('.unused'), config)
    outputs = [
        message['content'][0]['content']
        for message in result.messages
        if message['role'] == 'user'
    ]

    assert result.shortened_tool_results == 3
    assert outputs[0].startswith('[Older tool result omitted')
    assert outputs[-2:] == ['3' * 30, '4' * 30]


def test_middle_snip_never_splits_tool_use_and_result_pair(
    tmp_path: Path,
) -> None:
    messages: list[dict[str, object]] = [
        {'role': 'user', 'content': 'start'}
    ]
    for index in range(8):
        messages.extend(tool_pair(f'toolu_{index}', f'result {index}'))
    messages.append({'role': 'assistant', 'content': 'latest'})
    config = CompactionConfig(
        message_limit=8,
        keep_first_messages=3,
        keep_recent_messages=4,
    )

    result = cheap_compact(messages, tmp_path, config)
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in result.messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if block['type'] == 'tool_use':
                call_ids.add(block['id'])
            if block['type'] == 'tool_result':
                result_ids.add(block['tool_use_id'])

    assert call_ids == result_ids
    assert result.removed_messages > 0
    assert any(
        'middle messages' in str(message.get('content'))
        for message in result.messages
    )


def test_twenty_tool_rounds_remain_protocol_valid_and_bounded(
    tmp_path: Path,
) -> None:
    messages: list[dict[str, object]] = [
        {'role': 'user', 'content': 'long coding task'}
    ]
    for index in range(20):
        messages.extend(tool_pair(f'toolu_{index}', 'output ' + 'x' * 500))
    config = CompactionConfig(
        message_limit=16,
        keep_first_messages=3,
        keep_recent_messages=10,
        old_tool_result_limit=80,
        keep_recent_tool_results=3,
    )

    result = cheap_compact(messages, tmp_path, config)
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in result.messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if block['type'] == 'tool_use':
                call_ids.add(block['id'])
            elif block['type'] == 'tool_result':
                result_ids.add(block['tool_use_id'])

    assert call_ids == result_ids
    assert len(result.messages) < len(messages)
    assert result.shortened_tool_results > 0
