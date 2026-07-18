'''Tests for provider-neutral context accounting.'''

from forge.context.manager import ContextManager, context_stats


def test_context_stats_count_messages_and_tool_results() -> None:
    messages = [
        {'role': 'user', 'content': 'hello'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_1',
                    'content': 'result text',
                }
            ],
        },
    ]

    stats = context_stats(messages)

    assert stats.message_count == 2
    assert stats.tool_result_characters == len('result text')
    assert stats.estimated_characters > stats.tool_result_characters
    assert stats.estimated_tokens == (stats.estimated_characters + 3) // 4


def test_context_manager_reflects_mutated_history() -> None:
    messages: list[dict[str, object]] = []
    manager = ContextManager(messages)

    messages.append({'role': 'user', 'content': 'new message'})

    assert manager.stats.message_count == 1
