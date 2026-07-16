# TypeScript Todo Fixture

这是 ForgeCode M0 阶段的 TypeScript Bug 任务。项目实现一个内存 TodoList，并使用 TypeScript、Vitest 和 npm 锁文件提供可复现测试。

## 任务

修复 complete 根据数组下标而不是 Todo id 选择任务的问题，确保只完成指定 id 的任务，并补充或调整公开测试。

允许修改：

- src/
- tests/public/

禁止读取或修改：

- tests/hidden/

## 命令

安装锁定依赖、检查类型并运行公开测试：

    npm ci
    npm run build
    npm test

评测端在 Agent 结束后单独运行隐藏测试：

    npm run test:hidden

Fixture 的基础状态应有一个公开测试失败；正确修复后，构建、公开测试和隐藏测试都应通过。
