'''Regression checks for runtime responsibility boundaries.'''

import ast
from pathlib import Path


RUNTIME = Path(__file__).parents[2] / 'forge' / 'runtime'


def test_runtime_has_no_empty_python_modules() -> None:
    empty = [
        path.name
        for path in RUNTIME.glob('*.py')
        if not path.read_text(encoding='utf-8').strip()
    ]

    assert empty == []


def test_runtime_does_not_depend_on_shell_tool_implementation() -> None:
    offenders: list[str] = []
    for path in RUNTIME.glob('*.py'):
        if path.name == 'agent_loop.py':
            continue  # Composition root wires concrete tools.
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == 'forge.tools.shell'
            ):
                offenders.append(path.name)

    assert offenders == []


def test_shared_protocol_and_target_helpers_have_one_definition() -> None:
    expected = {
        'build_assistant_message': 'agent_messages.py',
        'build_tool_result_message': 'agent_messages.py',
        'mutation_target_paths': 'tool_targets.py',
    }
    definitions: dict[str, list[str]] = {name: [] for name in expected}
    roots = [RUNTIME, RUNTIME.parent / 'tools']
    for root in roots:
        for path in root.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in definitions:
                        definitions[node.name].append(path.name)

    assert definitions == {
        name: [filename] for name, filename in expected.items()
    }
