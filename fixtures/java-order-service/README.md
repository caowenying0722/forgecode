# Java Order Service Fixture

这是 ForgeCode M0 阶段的 Java Bug 任务。项目使用 Java 8、Maven 和 JUnit Jupiter 实现一个最小订单金额计算服务。

## 任务

修复订单总额计算忽略商品数量的问题，确保每个订单项按单价乘数量计入总额，并补充或调整公开测试。

允许修改：

- src/main/
- src/test/

禁止读取或修改：

- tests/hidden/

## 命令

构建并运行公开测试：

    mvn -q -DskipTests package
    mvn -q test

评测端在 Agent 结束后单独运行隐藏测试：

    mvn -q -Phidden-tests -Dtest=OrderServiceHiddenTest test

Fixture 的基础状态应有一个公开测试失败；正确修复后，构建、公开测试和隐藏测试都应通过。
