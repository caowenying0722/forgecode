# Python Calculator Fixture

这是 ForgeCode M0 阶段的第一个可复现 Bug 任务。项目包含一个小型四则运算库，以及公开测试和评测端保管的隐藏测试。

## 任务

修复除数为零时返回错误结果的问题，使 divide 与 Python 除法语义一致，并补充或调整公开测试。

允许修改：

- src/calculator/
- tests/public/

禁止读取或修改：

- tests/hidden/

## 命令

安装锁定依赖并运行公开测试：

    uv sync
    uv run pytest

评测端在 Agent 结束后单独运行隐藏测试：

    uv run pytest tests/hidden

Fixture 的基础状态应有一个公开测试失败；正确修复后，公开测试和隐藏测试都应通过。
