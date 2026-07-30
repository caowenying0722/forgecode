请直接在当前 ForgeCode 仓库中定位、修复并验证一个验证恢复状态机 Bug。不要只分析或给修改建议，要实际修改代码、补充回归测试并运行验证。

## Bug 复现背景

在一个空目录中让 ForgeCode 创建 Phaser + TypeScript 项目时，模型错误调用了：

```text
verify {
  "command": "npm install",
  "cwd": ".",
  "timeout_seconds": 600,
  "target": "build"
}
```

`npm install` 不是构建、测试、lint 或 typecheck 命令，因此 `verify` 正确返回：

```text
verification_status=invalid
exit_code=-1
Verification command is not a recognized non-interactive project validation command.
```

但之后 ForgeCode 出现错误恢复链：

1. 任意失败的 `verify` 都让控制器进入 `FIX_REQUIRED`。
2. `invalid` 因此被误认为“项目代码验证失败”。
3. Verification Recovery 隐藏 `verify`，要求模型先修改工作区。
4. 模型没有真实编译诊断，只能无目的修改项目文件。
5. 恢复提示说可以安装缺少的依赖或运行修复命令，但恢复工具集中没有 `run_command`。
6. 模型尝试以 `blocked` 结束，又被 completion checker 拒绝，因为这不是外部阻塞。
7. 多次完成声明失败后，任务最终进入 `stuck`：

```text
ForgeCode rejected the model completion declaration after repeated evidence failures.
```

## 首先检查这些文件

重点检查但不限于：

```text
forge/runtime/verification.py
forge/runtime/agent_controller.py
forge/runtime/request_builder.py
forge/runtime/recovery_manager.py
forge/runtime/completion_checker.py
forge/runtime/agent_loop.py

tests/runtime/test_verification.py
tests/runtime/test_agent_loop.py
tests/runtime/test_m2_agent_loop.py
tests/runtime/test_runtime_architecture.py
tests/runtime/test_workspace_completion.py
```

先沿着以下数据流确认根因，不要只根据本提示机械修改：

```text
VerifyTool 返回 ToolResult
→ verification_status 写入 metadata/evidence
→ AgentController.observe_tool_result
→ VerificationRecoveryState.requires_repair
→ RequestBuilder 选择恢复工具
→ RecoveryManager.verification_tools
→ CompletionChecker / finish_task
→ AgentLoop 的 stuck 判定
```

## 必须实现的行为

### 1. 按验证状态进行明确状态转换

不要再使用“所有 `verify` 失败都进入 `FIX_REQUIRED`”的逻辑。

至少保证：

```text
passed
→ 保持现有成功验证语义

failed
→ FIX_REQUIRED

timed_out
→ FIX_REQUIRED，除非项目当前已有更明确的超时恢复设计

invalid
→ 不得进入 FIX_REQUIRED
→ 不得要求先修改工作区
→ 应允许立即重新选择一个合法验证命令
→ 下一轮必须继续暴露 verify

unavailable
→ 单独审查并定义合理语义
→ 不要未经判断直接当成源码修复失败
```

建议把验证结果到控制状态的映射集中为可测试的显式逻辑，避免多个模块各自推断。

不要只判断 `result.success`，应读取并规范处理：

```python
result.metadata["verification_status"]
```

需要为缺失或未知状态提供保守的兼容处理，但不能让 `invalid` 回落到 `FIX_REQUIRED`。

### 2. `requires_repair()` 只能表示真实的项目验证失败

修正 `VerificationRecoveryState.requires_repair()`，使它不能仅根据：

```python
latest is not None and not latest.success
```

来判断。

它应该只对真正要求修改工作区的状态返回 `True`，例如：

```text
failed
timed_out
```

`invalid` 不代表当前 revision 的代码、测试或构建失败。

### 3. 无效验证命令后应允许立即重试

增加端到端或接近端到端的回归测试，证明以下流程成立：

```text
第一次 verify("npm install")
→ 返回 invalid
→ 工作区 revision 不变
→ 不要求任何文件修改
→ 下一模型请求仍包含 verify
→ 可调用 verify(target="auto") 或发现到的合法验证命令
→ 合法验证成功后可以正常 finish_task
→ 任务不会进入 stuck
```

测试必须检查实际暴露的工具名称，而不只是测试单个辅助函数。

### 4. 保留真实验证失败的修复门禁

增加测试确认此次修复没有削弱现有安全约束：

```text
合法 build/test/typecheck 命令实际执行并失败
→ 仍进入 FIX_REQUIRED
→ 在相关工作区修复前，不允许把旧 revision 声明为已验证
→ 产生新的相关 workspace revision 后，verify 再次可用
```

不要为了修复 `invalid` 而让所有失败都可以立即重复验证。

### 5. 解决 Verification Recovery 与 `run_command` 的矛盾

当前恢复提示允许模型：

```text
install missing dependencies
run the concrete repair command
```

但 `RecoveryManager.verification_tools()` 在 `fix_available=True` 时只提供读取和写入类工具，没有 `run_command`。

请选择一个一致且最小的方案，优先考虑：

```text
在真实验证失败的修复恢复阶段暴露 run_command
```

要求：

* 仍沿用现有命令权限和安全检查。
* 不绕过 ToolRunner 的 effect/permission 机制。
* 不把任意进程工具全部开放。
* 仅加入完成验证修复确实需要的 `run_command`。
* 增加测试确认 `fix_available=True` 时工具面包含 `run_command`。
* 增加测试确认无关的危险或被排除工具没有因此暴露。

如果仓库架构明确禁止在该阶段暴露 `run_command`，则应修改恢复提示，使其不再要求模型执行它无法调用的操作，并提供一条不会死锁的替代路径。但必须通过测试证明缺少依赖时仍能继续推进；不能只改提示文本掩盖问题。

### 6. 不要放宽 `blocked` 的定义

不要通过把 ForgeCode 自己的恢复状态错误包装成外部 blocker 来解决问题。

以下语义应继续保留：

```text
blocked 仅用于需要用户操作、权限、凭据或不可用外部依赖的情况。
```

这个 Bug 应通过正确的验证状态转换和工具可用性修复，而不是放宽完成门禁。

增加或保留测试，确认普通的命令参数错误、无效验证命令或无进展不能声明为 `blocked`。

### 7. 不要生成虚假的 Repair Target

审查：

```text
repair_target_from_verification
repair_target_from_verification_result
```

对于 `verification_status=invalid`：

* 不应基于当前 changed paths 生成“修改源码”的 Repair Target。
* 不应消耗针对源码修复的读取预算。
* 恢复提示应明确告诉模型重新选择合法验证命令。
* 不要提示模型编辑与错误无关的文件。

## 建议添加的回归测试

测试名称可以按现有风格调整，但至少覆盖同等行为：

```text
test_invalid_verify_does_not_enter_fix_required
test_invalid_verify_keeps_verify_available
test_invalid_verify_does_not_require_workspace_revision
test_invalid_verify_can_retry_with_auto_target
test_invalid_verify_does_not_create_source_repair_target
test_failed_verification_still_requires_repair
test_verification_recovery_exposes_run_command_for_real_repair
test_blocked_still_requires_external_condition
test_invalid_then_valid_verification_can_finish_without_stuck
```

优先复用现有 fixture、FakeModelClient、ToolResult builder 和请求工具面断言，不要新建一套重复测试框架。

## 实现约束

* 先阅读现有状态机和测试，再修改。
* 做最小、内聚的修改，不要大规模重构 Agent Loop。
* 不要通过增加重试次数、提高 token 上限或取消完成门禁规避问题。
* 不要硬编码 `npm install`。
* 修复应适用于所有被分类为 `verification_status=invalid` 的命令。
* 不要把 `invalid` 改成成功验证；它仍然是一次失败的工具调用，只是不代表工作区需要修复。
* 保持已有公开 API 和序列化数据尽量兼容。
* 若修改状态枚举或 metadata 结构，必须补兼容测试。
* 不要修改与该 Bug 无关的项目代码。
* 不要只测试私有辅助函数；必须至少有一条覆盖请求工具面或 Agent Loop 的行为测试。

## 验证要求

完成修改后，依次运行与仓库当前配置相符的验证命令。至少运行：

```bash
uv lock --check
uv run python -m compileall -q forge tests
uv run pytest -q
git diff --check
```

先运行相关的定向测试以快速定位失败，再运行完整测试集。

如果完整测试失败：

1. 判断是否由本次修改引起。
2. 修复本次引入的回归。
3. 不得在测试未通过时声称完成。

## 最终输出格式

完成后给出：

1. 根因说明。
2. 修改了哪些文件。
3. 每个关键修改解决了哪一段错误状态转换。
4. 新增了哪些回归测试。
5. 实际执行的验证命令及结果。
6. 是否仍存在已知边界情况。
7. 简洁的 `git diff --stat` 摘要。

请现在直接检查仓库、复现问题、修改代码、运行测试并完成修复。
