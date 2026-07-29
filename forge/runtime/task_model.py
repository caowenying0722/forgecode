'''Model-visible task model and minimal-context guidance.'''

from __future__ import annotations

from dataclasses import dataclass

from forge.runtime.intent import TaskContract


@dataclass(frozen=True, slots=True)
class RuntimeTaskModel:
    '''Structured execution target derived before the first model request.'''

    goal: str
    objective: str
    expected_steps: tuple[str, ...]
    completion_conditions: tuple[str, ...]
    context_plan: tuple[str, ...]
    scope_patterns: tuple[str, ...] = ()


def build_runtime_task_model(
    goal: str,
    contract: TaskContract,
    *,
    scope_patterns: tuple[str, ...] = (),
) -> RuntimeTaskModel:
    '''Build deterministic task guidance from the intent contract and scope.'''
    clean_goal = goal.strip()
    if contract.requires_change:
        return RuntimeTaskModel(
            goal=clean_goal,
            objective='Implement the requested workspace change.',
            expected_steps=(
                'Identify the smallest relevant code area.',
                'Read only the files or ranges needed to make the edit.',
                'Modify task-relevant files.',
                'Run the strongest available non-interactive verification.',
                'Finish only when the diff, verification, and summary match the goal.',
            ),
            completion_conditions=(
                'A task-local workspace diff exists.',
                'Changed paths are relevant to the user goal.',
                'Verification is current for the final workspace revision.',
            ),
            context_plan=(
                'Start from explicit paths in the prompt or known repository context.',
                'If no target path is known, use one narrow search for a domain term.',
                'After a likely target file is found, prefer focused read_file ranges.',
                'Do not rescan the repository after enough evidence exists to edit.',
            ),
            scope_patterns=scope_patterns,
        )
    if contract.requires_inspection_evidence:
        return RuntimeTaskModel(
            goal=clean_goal,
            objective='Answer from repository evidence without editing files.',
            expected_steps=(
                'Collect the minimum evidence needed for the question.',
                'Stop reading once the answer is supported.',
                'Summarize with concrete file or repository evidence.',
            ),
            completion_conditions=(
                'The answer is grounded in repository evidence.',
                'No workspace edit is made.',
            ),
            context_plan=(
                'Use read-only tools only.',
                'Prefer exact paths or one narrow search over broad directory scans.',
            ),
            scope_patterns=scope_patterns,
        )
    return RuntimeTaskModel(
        goal=clean_goal,
        objective='Respond directly unless the contract exposes read-only tools.',
        expected_steps=(
            'Use the inferred intent to decide whether tools are necessary.',
            'Avoid repository exploration when a direct answer is sufficient.',
        ),
        completion_conditions=(
            'The response matches the user intent.',
            'No unnecessary workspace edit is made.',
        ),
        context_plan=(
            'Do not load repository context unless it is needed for the answer.',
        ),
        scope_patterns=scope_patterns,
    )


def render_runtime_task_model(model: RuntimeTaskModel) -> str:
    '''Render the task model as concise system context.'''
    lines = [
        '[ForgeCode Runtime Task Model]',
        f'Objective: {model.objective}',
        'Expected steps:',
        *[f'- {step}' for step in model.expected_steps],
        'Completion conditions:',
        *[f'- {condition}' for condition in model.completion_conditions],
        'Context plan:',
        *[f'- {item}' for item in model.context_plan],
    ]
    if model.scope_patterns:
        patterns = ', '.join(model.scope_patterns[:16])
        suffix = ' ...' if len(model.scope_patterns) > 16 else ''
        lines.extend(
            [
                'Expected impact scope:',
                f'- {patterns}{suffix}',
            ]
        )
    return '\n'.join(lines)
