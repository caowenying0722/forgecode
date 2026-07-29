'''Tests for conservative workspace-change intent inference.'''

import pytest

from forge.runtime.intent import infer_change_required, infer_task_contract


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
        '可以，开始吧',
        '阅读当前目录下的任务文件task.md，明确任务后开始工作',
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
