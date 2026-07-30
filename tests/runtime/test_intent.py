'''Tests for conservative workspace-change intent inference.'''

import asyncio
from dataclasses import replace

import pytest

from forge.runtime.intent import (
    TaskContract,
    TurnIntent,
    infer_change_required,
    infer_task_contract,
    refine_task_contract,
    refine_task_contract_async,
)


@pytest.mark.parametrize(
    'prompt',
    [
        '当前游戏有一个 bug，帮我修复一下',
        '请修改 README.md',
        '优化一下当前的上下文管理',
        '帮我在配置文件中添加一个开关',
        '帮我解决这个 bug',
        '帮我改一下',
        '请检查并修复这个 bug',
        '按刚才的方案执行',
        '按最高优先级 P0 进行修复',
        '把 world.js 改成六面渲染',
        '那就直接加入暂停功能',
        '阅读当前目录下的任务文件task.md，明确任务后开始工作',
        '根据 task.md 继续完善项目功能',
        '根据task.md完善整个项目',
        '按照需求继续开发这个游戏项目',
        'Fix the rendering bug.',
        'Please resolve the rendering bug.',
        'Inspect and fix the rendering bug.',
        'Could you please update the CLI?',
        'Help me implement streaming output.',
        'Make a real code change',
        '帮我新建一个 game 目录',
    ],
)
def test_explicit_change_requests_require_a_workspace_diff(
    prompt: str,
) -> None:
    assert infer_change_required(prompt) is True


@pytest.mark.parametrize(
    'prompt',
    [
        '为什么会出现这个 bug？',
        '如何修复这个问题？',
        '帮我解释如何修改 README',
        '给出一个修复方案，我再决定是否执行',
        '好的，帮我按“优先级P0/P1/P2”给你出一版最小改动修复清单',
        '按 P0/P1/P2 列一个最小改动修复 checklist',
        '完成了吗？',
        '优化方案是什么？',
        '修改方案是什么？',
        '优化建议有哪些？',
        '更新一下当前进度',
        '为什么你不能帮我修改文件？',
        '帮我优化这个方案，不要修改代码',
        '继续解释刚才的实现思路',
        '查看 play 目录',
        'Explain how to fix the rendering bug.',
        'Update me on the current progress.',
        'Write a plan for the refactor.',
        'Give me a P0/P1/P2 fix checklist.',
        'Plan a refactor, but do not change files.',
        '可以，开始吧',
    ],
)
def test_questions_and_plans_do_not_require_a_workspace_diff(
    prompt: str,
) -> None:
    assert infer_change_required(prompt) is False


def test_auto_fix_prompt_creates_change_contract() -> None:
    contract = infer_task_contract(
        '继续补齐当前项目的核心游戏骨架',
        workspace_available=True,
    )

    assert contract.intent.kind == 'implement'
    assert contract.requires_change is True
    assert contract.completion_contract == 'change'
    assert contract.initial_phase == 'implementing'
    assert contract.initial_tool_surface == 'all'


@pytest.mark.parametrize(
    'prompt',
    [
        '@task.md 阅读任务文档，在当前目录下实现',
        '阅读 task.md，在当前项目实现',
        '查看需求文档，完成其中要求',
        '阅读说明并开始落地',
        '先查看 task.md，然后创建完整项目',
    ],
)
def test_read_task_document_then_implement_is_change_intent(
    prompt: str,
) -> None:
    contract = infer_task_contract(prompt, workspace_available=True)

    assert contract.requires_change is True
    assert contract.completion_contract == 'change'
    assert contract.initial_tool_surface == 'all'


def test_read_task_document_in_current_directory_is_change_intent() -> None:
    contract = infer_task_contract(
        '@task.md 阅读任务文档，在当前目录下实现',
        workspace_available=True,
    )

    assert contract.requires_change is True
    assert contract.initial_tool_surface == 'all'
    assert contract.allowed_paths == ()
    assert contract.context_hints == ('task.md',)


def test_read_task_document_and_summarize_remains_inspection() -> None:
    contract = infer_task_contract(
        '阅读 task.md 并总结',
        workspace_available=True,
    )

    assert contract.requires_change is False
    assert contract.initial_tool_surface == 'read_only'


def test_task_document_implementation_plan_remains_advisory() -> None:
    contract = infer_task_contract(
        '阅读任务文档并给出实现方案',
        workspace_available=True,
    )

    assert contract.requires_change is False
    assert contract.requires_plan is True
    assert contract.initial_tool_surface == 'read_only'


def test_negated_task_document_change_remains_read_only() -> None:
    contract = infer_task_contract(
        '阅读 task.md，不要修改文件',
        workspace_available=True,
    )

    assert contract.requires_change is False
    assert contract.initial_tool_surface == 'read_only'


def test_plan_prompt_creates_read_only_contract() -> None:
    contract = infer_task_contract(
        '给出一个修复方案，我再决定是否执行',
        workspace_available=True,
    )

    assert contract.intent.kind == 'plan'
    assert contract.requires_change is False
    assert contract.requires_plan is True
    assert contract.initial_tool_surface == 'read_only'


def test_explicit_modes_override_prompt_intent() -> None:
    plan = infer_task_contract(
        '帮我修复 bug',
        interaction_mode='plan',
        workspace_available=True,
    )
    code = infer_task_contract(
        '给我一个计划',
        interaction_mode='code',
        workspace_available=True,
    )

    assert plan.requires_change is False
    assert plan.initial_tool_surface == 'read_only'
    assert code.requires_change is True
    assert code.initial_tool_surface == 'all'


@pytest.mark.parametrize(
    'prompt',
    [
        '还有加入其他功能让整个项目更完善吗？',
        '你建议下一步做什么？',
        '这个项目还能增加哪些功能？',
        '给我几个优化方向。',
    ],
)
def test_global_policy_does_not_turn_advice_into_change(
    prompt: str,
) -> None:
    contract = infer_task_contract(
        prompt,
        workspace_available=True,
        policy_requires_change=True,
    )

    assert contract.kind == 'advisory'
    assert contract.requires_change is False
    assert contract.completion_contract == 'none'
    assert contract.initial_tool_surface == 'read_only'


def test_code_mode_can_force_implementation_contract() -> None:
    contract = infer_task_contract(
        '还有加入其他功能让整个项目更完善吗？',
        interaction_mode='code',
        workspace_available=True,
        policy_requires_change=False,
    )

    assert contract.kind == 'implement'
    assert contract.requires_change is True


@pytest.mark.parametrize(
    ('prompt', 'expected_kind', 'tool_surface', 'requires_change'),
    [
        (
            '解释一下 Python 里的生成器是什么',
            'answer',
            'none',
            False,
        ),
        (
            '分析 forge/runtime/intent.py 的职责',
            'inspect',
            'read_only',
            False,
        ),
        (
            '只分析 forge/runtime/intent.py，不要修改',
            'inspect',
            'read_only',
            False,
        ),
        (
            '厦门今天天气',
            'inspect',
            'read_only',
            False,
        ),
        (
            '联网查询厦门今天天气',
            'inspect',
            'read_only',
            False,
        ),
        (
            'Search the latest Python release',
            'inspect',
            'read_only',
            False,
        ),
        (
            '请重构多个模块的架构并更新相关代码',
            'refactor',
            'all',
            True,
        ),
    ],
)
def test_task_envelope_routes_obvious_requests(
    prompt: str,
    expected_kind: str,
    tool_surface: str,
    requires_change: bool,
) -> None:
    contract = infer_task_contract(prompt, workspace_available=True)

    assert contract.kind == expected_kind
    assert contract.goal == prompt
    assert contract.initial_tool_surface == tool_surface
    assert contract.requires_change is requires_change
    assert contract.confidence in {'medium', 'high'} or expected_kind == 'answer'


def test_task_envelope_records_paths_and_acceptance() -> None:
    contract = infer_task_contract(
        '分析 forge/runtime/intent.py 的职责',
        workspace_available=True,
    )

    assert contract.allowed_paths == ('forge/runtime/intent.py',)
    assert contract.context_hints == ('forge/runtime/intent.py',)
    assert contract.completion_contract == 'inspection'
    assert 'No workspace edit is made.' in contract.acceptance_criteria


def test_multi_module_architecture_change_requires_plan() -> None:
    contract = infer_task_contract(
        '请重构多个模块的架构并更新相关代码',
        workspace_available=True,
    )

    assert contract.requires_change is True
    assert contract.requires_plan is True
    assert contract.verification_policy.required is True


def test_semantic_classifier_is_used_only_for_low_confidence() -> None:
    class FakeClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def classify_task(
            self,
            prompt: str,
            baseline: TaskContract,
        ) -> TaskContract:
            self.calls += 1
            return replace(
                baseline,
                intent=TurnIntent('inspect', 'medium', f'classified:{prompt}'),
                kind='inspect',
                confidence='medium',
                semantic_classification='not_needed',
            )

    classifier = FakeClassifier()
    high = infer_task_contract('分析 forge/runtime/intent.py')
    low = infer_task_contract('生成器是什么')

    assert refine_task_contract('分析 forge/runtime/intent.py', high, classifier) is high
    refined = refine_task_contract('生成器是什么', low, classifier)

    assert classifier.calls == 1
    assert refined.kind == 'inspect'


def test_semantic_classifier_receives_history_for_ambiguous_followup() -> None:
    class ContextAwareClassifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def classify_task(
            self,
            prompt: str,
            baseline: TaskContract,
            semantic_context: str = '',
        ) -> TaskContract:
            self.calls.append((prompt, semantic_context))
            return replace(
                baseline,
                intent=TurnIntent('implement', 'medium', 'classified follow-up'),
                kind='implement',
                requires_change=True,
                completion_contract='change',
                initial_tool_surface='all',
                semantic_classification='recommended',
            )

    baseline = infer_task_contract('可以，开始吧')
    classifier = ContextAwareClassifier()

    refined = asyncio.run(
        refine_task_contract_async(
            '可以，开始吧',
            baseline,
            classifier,
            semantic_context='Active task: add pause menu',
        )
    )

    assert classifier.calls == [('可以，开始吧', 'Active task: add pause menu')]
    assert refined.requires_change is True
    assert refined.kind == 'implement'
