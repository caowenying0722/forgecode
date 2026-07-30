请修复“成功构建产生的合法验证产物被 CompletionChecker 当成任务外垃圾文件，导致任务无法完成”的 Bug。

不要只做分析或给出伪代码。请检查当前实现、添加失败回归测试、修改源码、运行定向测试和完整测试，并报告真实结果。

## Bug 复现

用户要求 ForgeCode 根据 `task.md` 创建一个 Vite + TypeScript + Phaser 项目。

ForgeCode 已经成功：

```text
- 创建 package.json、tsconfig.json、vite.config.ts、index.html
- 创建 src/** 和 tests/**
- npm test 通过
- verify target=build 通过
```

实际构建命令：

```bash
npx tsc --noEmit -p tsconfig.json && npx vite build
```

Vite 正常生成：

```text
dist/index.html
dist/assets/index-4ZTasLJF.js
```

但完成检查随后拒绝：

```text
The workspace also contains generated, cache, or temporary files outside
the task deliverables:
dist/assets/index-4ZTasLJF.js, dist/index.html
```

模型删除 `dist/` 后再次执行 build，Vite 又重新生成 `dist/`，CompletionChecker 又再次拒绝，最后任务进入：

```text
task stuck
```

这是一个不可满足的循环：

```text
要证明 build 成功
→ 必须执行 vite build
→ vite build 必然生成 dist/**

要通过 completion
→ 当前逻辑又要求 dist/** 不存在
```

## 已知实现不一致

请先沿源码自行确认，不能只机械照抄本提示。

当前实现大致存在以下链路：

1. `verification_artifact_scope()` 对 Vite build 声明：

```python
ArtifactRule(
    "dist/**",
    "generated_artifact",
    "vite build output",
)
```

2. `VerifyTool` 使用：

```python
tracker.refresh(
    origin="verification",
    artifact_scope=scope,
)
```

3. 验证结果 metadata 会记录：

```text
verification_artifact_scope
generated_artifact_paths
cache_paths
verification_side_effect_paths
source_revision
filesystem_revision
```

4. `VerificationEvidence` 已经定义：

```python
generated_artifact_paths
cache_paths
verification_side_effect_paths
```

5. 但 `VerificationLedger.VerificationRecord` 当前只保存：

```text
artifact_scope
side_effect_paths
```

没有保存：

```text
generated_artifact_paths
cache_paths
```

6. `VerificationRecord.to_evidence()` 也没有恢复这些字段。

7. 因此，通过 Verification Ledger 取出的最新证据会丢失合法构建产物信息。

8. `CompletionChecker._with_relevance_reasons()` 当前没有接收 `verification` 参数，而是把：

```python
tracker.filesystem_changed_paths - tracker.changed_paths
```

重新分类，并对所有：

```text
generated_paths
cache_paths
unrelated_paths
```

统一添加阻塞原因。

9. 该逻辑没有区分：

```text
由当前成功验证明确声明的合法产物
```

和：

```text
模型自行创建的无关生成物、缓存或临时文件
```

最终导致已经通过 build 的任务被错误标记为 `stuck`。

## 重点检查文件

请重点检查但不限于：

```text
forge/runtime/verification.py
forge/tools/verify.py
forge/runtime/state.py
forge/runtime/verification_ledger.py
forge/runtime/workspace.py
forge/runtime/workspace_classification.py
forge/runtime/completion.py
forge/runtime/completion_checker.py
forge/runtime/agent_loop.py
forge/runtime/task_scope.py
```

相关测试优先放入现有文件：

```text
tests/runtime/test_verification.py
tests/runtime/test_workspace_completion.py
tests/runtime/test_agent_loop.py
tests/runtime/test_m2_agent_loop.py
```

请搜索并追踪：

```text
VerificationArtifactScope
ArtifactRule
verification_artifact_scope
generated_artifact_paths
cache_paths
verification_side_effect_paths
VerificationEvidence
VerificationRecord
VerificationLedger.record_from_metadata
VerificationRecord.to_evidence
verification_from_result
WorkspaceTracker._artifact_paths
WorkspaceTracker.filesystem_changed_paths
WorkspaceTracker.changed_paths
CompletionChecker._with_relevance_reasons
CompletionGate.evaluate
can_finalize_after_stagnation
finish_rejection_reasons
```

## 核心修复目标

### 1. 完整保留验证产物证据

修复 `VerificationLedger` 的数据丢失。

`VerificationRecord` 至少需要保存：

```python
generated_artifact_paths: tuple[str, ...] = ()
cache_paths: tuple[str, ...] = ()
```

`record_from_metadata()` 必须从 ToolResult metadata 中读取：

```text
generated_artifact_paths
cache_paths
```

`to_evidence()` 必须恢复到：

```python
VerificationEvidence(
    ...
    generated_artifact_paths=self.generated_artifact_paths,
    cache_paths=self.cache_paths,
    verification_side_effect_paths=self.side_effect_paths,
)
```

还要检查：

```text
- verify 直接返回路径
- Verification Ledger 路径
- reusable verification cache 路径
- verification_from_result 路径
```

确保无论 Agent Loop 最终从哪条路径获得 `VerificationEvidence`，这些字段都不会丢失。

不得只修改 `VerificationEvidence`，因为该类型现在已经有这些字段；需要修复完整的序列化和恢复链路。

### 2. CompletionChecker 必须接收并使用当前验证证据

修改：

```python
CompletionChecker._with_relevance_reasons(...)
```

使它能够使用当前：

```python
verification: VerificationEvidence | None
```

更新所有调用位置，包括但不限于：

```text
CompletionChecker.evaluate
finish_rejection_reasons
can_finalize_after_stagnation
其他直接调用 _with_relevance_reasons 的位置
```

不要通过读取某个全局变量或重新解析日志获取验证状态。

应从权威的结构化 `VerificationEvidence` 中判断合法产物。

### 3. 定义“可信验证产物”

新增一个集中、可测试的 helper，名称可按项目风格调整，例如：

```python
trusted_verification_output_paths(...)
verified_artifact_paths_for_current_workspace(...)
is_current_verified_output(...)
```

一条路径只有同时满足以下条件时，才可以作为可信验证产物，不阻塞完成：

```text
1. verification 不为 None
2. verification.success 为 True
3. verification.bound_source_revision == tracker.source_revision
4. verification.verification_side_effect_paths 为空
5. path 明确出现在：
   - verification.generated_artifact_paths
   - 或 verification.cache_paths
6. path 当前仍属于 tracker.filesystem_changed_paths
7. 路径没有命中 forbidden policy
```

随后计算：

```text
extra filesystem paths
- trusted verification outputs
= 真正需要阻塞的额外路径
```

本次场景中：

```text
dist/index.html
dist/assets/index-4ZTasLJF.js
```

由当前 source revision 的成功 Vite build 生成，并明确包含于 `generated_artifact_paths`，所以不应阻塞任务完成。

### 4. 防止验证后篡改产物被错误信任

不能只根据：

```text
path 出现在 generated_artifact_paths
+
source_revision 没变化
```

就永久信任路径。

原因是模型可能在验证完成后手工修改：

```text
dist/index.html
```

这类产物修改通常不会增加 `source_revision`，如果只比较 source revision，就可能把验证后的手工改动错误视为可信。

请实现一个确定性的完整性保护方案。

优先方案：

在验证完成时记录合法验证产物的内容指纹，例如：

```python
generated_artifact_fingerprints: tuple[tuple[str, str], ...] = ()
cache_fingerprints: tuple[tuple[str, str], ...] = ()
```

或者使用明确的不可变映射类型。

要求：

```text
- VerifyTool 在验证后从 WorkspaceTracker 当前 snapshot 获取指纹
- metadata 保存路径和指纹
- VerificationRecord 保存并恢复指纹
- VerificationEvidence 保存指纹
- CompletionChecker 只信任当前指纹与验证时一致的产物
```

若仓库已有可复用的 snapshot/fingerprint API，请复用，避免重复实现文件 Hash 逻辑。

可以采用其他同等安全的设计，但必须证明：

```text
验证后手工修改 dist/** 不会继续被信任
```

仅要求：

```python
verification.filesystem_revision == tracker.filesystem_revision
```

可能过于严格，因为验证后删除一个无关缓存、清理锁文件或执行不影响源码的操作也会增加 filesystem revision。若采用 filesystem revision 方案，必须通过测试证明不会重新制造清理和重验循环。

### 5. 合法验证产物不能作为任务进展

本次修复只能让合法产物“不阻塞完成”，不能让它们成为 Change 任务的实现证据。

必须保留：

```text
只有 dist/** 发生变化
且没有源码、测试、配置或任务支持文件变化
→ 仍然不能满足 change task
```

也就是说：

```python
tracker.changed_paths
```

仍然必须代表真实任务交付物。

可信验证产物只能从“额外路径阻塞检查”中排除，不能加入：

```text
changed_paths
source_revision
task deliverables
acceptance evidence
```

### 6. 未声明产物仍然必须阻塞

下面的情况不得因本次修复而放行：

```text
验证命令声明只允许 dist/**
但实际额外生成 tmp/debug.log
→ tmp/debug.log 仍然阻塞

验证命令修改 src/main.ts
→ verification_side_effect
→ 验证失败并阻塞

模型自行创建 dist/manual.js
且该文件不是本次验证产生或指纹不匹配
→ 仍然阻塞

验证失败但留下 dist/**
→ 不信任这些产物

验证超时但留下 build/**
→ 不信任这些产物

源码在 build 后再次修改
→ 旧 build 证据 stale
→ 不能完成

当前验证属于旧 source revision
→ 不能用其产物白名单完成
```

不要简单地在 CompletionChecker 中永久忽略：

```text
dist/**
build/**
coverage/**
```

也不要把所有 generic generated paths 都放行。

### 7. 保持 forbidden path 优先级

即使某个验证 Scope 错误声明了禁止路径，也不能绕过：

```text
TaskPolicy.forbidden_paths
DEFAULT_FORBIDDEN_PATTERNS
workspace root 安全边界
```

例如：

```text
tests/hidden/**
../outside
绝对路径
```

永远不能因为出现在 `generated_artifact_paths` 中就被信任。

可信验证产物过滤必须发生在 forbidden 检查之后，或显式排除 forbidden paths。

### 8. 清理策略不要与允许策略混淆

当前 `VerificationArtifactScope` 有：

```python
cleanup_generated: bool = False
```

请检查该字段是否已真正使用。

此次优先采用：

```text
允许当前成功验证声明的合法产物留在工作区
```

而不是贸然开启自动删除。

原因是自动清理需要处理：

```text
- build 前已经存在的 dist/
- 用户原有构建产物
- 自定义 outDir
- 验证覆盖的旧文件
- 验证新增文件
- 删除后的验证证据是否仍有效
```

除非仓库已经有完整、安全、经过测试的产物恢复机制，否则不要仅把：

```python
cleanup_generated=True
```

作为表面修复。

可以保留该字段，或补充注释说明此次策略，但不要构建一套未经验证的删除逻辑。

### 9. 支持自定义 Vite outDir

当前 `_vite_output_rules()` 会尝试从：

```text
vite.config.ts
vite.config.js
vite.config.mjs
```

提取：

```typescript
build: {
  outDir: "release"
}
```

本次完成逻辑不能硬编码 `dist/**`。

以下场景也应通过：

```text
vite build → release/index.html
verification.generated_artifact_paths 包含 release/index.html
当前 source revision 验证成功
→ completion 不应阻塞
```

生产代码不得硬编码本次 Hash 文件名。

### 10. 避免删除后再次 build 的循环

请增加接近 Agent Loop 的测试，证明：

```text
1. 修改真实源码
2. verify build 成功
3. build 生成合法输出
4. CompletionChecker 允许完成
5. Agent Loop 不要求清理 dist/
6. 不会触发“删除 dist → 重新 build → 再生成 dist”的循环
7. TurnResult.status == completed
8. 不会进入 stuck
```

修复不能只让一个私有 helper 的单元测试通过。

## 次要问题：package-lock.json

日志中早期还出现：

```text
The workspace changed, but these paths are outside the task scope or are
forbidden: package-lock.json
```

`package-lock.json` 在 WorkspaceChangeClassifier 中属于项目配置文件，但显式任务 Scope 可能只列出了：

```text
package.json
tsconfig.json
vite.config.ts
index.html
src/**
tests/**
```

请在完成主要 Bug 后审查这一行为。

对于：

```text
从零创建 npm 项目
+
运行 npm install
```

`package-lock.json` 通常是正常的项目支持配置，不应迫使模型删除锁文件。

优先考虑一个小而通用的规则：

```text
当 package.json 在任务范围中，并且本轮合法安装了 npm 依赖时，
package-lock.json 作为配套配置路径可以被视为 supporting path。
```

同理可考虑：

```text
pnpm-lock.yaml
yarn.lock
```

但不要让存在 `package.json` 自动放开工作区所有文件。

这一项不是最终 `dist/**` stuck 的直接原因，不能因处理锁文件而偏离主要修复。

## 推荐数据流

建议最终形成：

```text
verify command starts
→ derive VerificationArtifactScope
→ execute command
→ tracker.refresh(origin="verification", artifact_scope=scope)
→ classify:
   - generated artifacts
   - cache
   - verification side effects
→ capture artifact fingerprints
→ ToolResult metadata
→ VerificationLedger.record_from_metadata
→ VerificationRecord
→ VerificationEvidence
→ CompletionChecker
→ exclude only current, successful, fingerprint-matching verified outputs
→ continue rejecting all other generated/cache/unrelated paths
```

不要在 CompletionChecker 中重新猜测某个文件是否由 Vite、Webpack 或其他构建器生成。

验证适配器负责声明输出，VerifyTool 负责记录事实，CompletionChecker 只消费结构化证据。

## 必须添加的测试

测试名称可以按现有命名风格调整，但必须覆盖等价行为。

### Verification Ledger 测试

```text
test_verification_ledger_preserves_generated_artifact_paths
test_verification_ledger_preserves_cache_paths
test_verification_ledger_preserves_artifact_fingerprints
test_cached_verification_preserves_artifact_evidence
```

核心断言：

```python
record = ledger.record_from_metadata(...)
evidence = record.to_evidence()

assert evidence.generated_artifact_paths == (
    "dist/index.html",
    "dist/assets/app.js",
)
assert evidence.cache_paths == (...)
```

### CompletionChecker 测试

```text
test_current_successful_vite_build_artifacts_do_not_block_completion
test_generated_artifacts_alone_do_not_satisfy_change_task
test_failed_build_artifacts_still_block_completion
test_timed_out_build_artifacts_still_block_completion
test_stale_build_artifacts_are_not_trusted
test_undeclared_generated_output_still_blocks_completion
test_verification_source_side_effect_still_blocks_completion
test_forbidden_path_is_never_trusted_as_verification_output
```

### 产物完整性测试

```text
test_verified_artifact_modified_after_build_is_not_trusted
test_verified_artifact_unchanged_after_build_is_trusted
test_deleted_verified_artifact_does_not_block_completion
```

场景：

```text
verify 生成 dist/index.html
→ 保存指纹
→ 模型随后修改 dist/index.html
→ 当前指纹不同
→ completion 必须拒绝或要求重新验证/清理
```

### 自定义 outDir 测试

```text
test_custom_vite_out_dir_is_allowed_as_verified_output
```

配置：

```typescript
export default defineConfig({
  build: {
    outDir: "release"
  }
});
```

验证后：

```text
release/index.html
release/assets/app.js
```

不应阻塞完成。

### Agent Loop 回归测试

完整模拟：

```text
1. 模型修改 src/main.ts
2. verify target=build
3. fake build 创建 dist/index.html 和 dist/assets/app.js
4. verify 返回 exit 0
5. 当前 source revision 未改变
6. 模型给出完成声明
7. status == completed
8. completion_reasons 为空
9. 不要求删除 dist/
10. 不再次请求 verify
11. 不进入 stuck
```

建议名称：

```text
test_successful_build_with_declared_outputs_finishes_without_cleanup_loop
```

### 反例：只有构建产物

```text
初始源码没有变化
仅执行 build 生成 dist/**
→ completion 不得将其视为真实任务修改
```

建议名称：

```text
test_build_outputs_alone_do_not_count_as_task_progress
```

### Lockfile 测试

若同时修复锁文件支持：

```text
test_new_npm_project_accepts_package_lock_as_supporting_config
test_lockfile_does_not_expand_scope_to_unrelated_paths
```

## 兼容性要求

* 保持 `VerificationEvidence` 现有构造调用兼容，新字段必须有默认值。
* 保持已有 Verification Ledger 记录兼容。
* 不要删除现有 metadata 字段。
* 不要要求调用方同时维护两套产物列表。
* 不要通过解析验证 stdout 推断产物。
* 不要硬编码 Vite 作为唯一框架。
* 不要禁用 WorkspaceTracker 对 ignored artifacts 的追踪。
* 不要把生成物加入 source revision。
* 不要放宽 verification side-effect 检查。
* 不要放宽 forbidden paths。
* 不要提高 stuck 或 completion retry 上限来掩盖问题。
* 不要让模型必须执行“最后 build 后手动删除产物”的特殊流程。
* 不要删除、跳过或弱化已有测试来制造通过。

## 推荐实现顺序

1. 先添加复现当前日志的失败测试。
2. 修复 Verification Ledger 对产物字段的数据丢失。
3. 增加产物指纹或同等完整性绑定。
4. 让 CompletionChecker 接收当前 VerificationEvidence。
5. 集中实现可信验证产物过滤。
6. 保留未声明、失败、过期和被篡改产物的拒绝逻辑。
7. 增加自定义 outDir 测试。
8. 增加 Agent Loop 完整回归测试。
9. 审查并按最小范围处理 package-lock.json。
10. 运行定向测试和完整测试。

## 验证命令

先运行定向测试：

```bash
uv run pytest -q tests/runtime/test_verification.py
uv run pytest -q tests/runtime/test_workspace_completion.py
uv run pytest -q tests/runtime/test_agent_loop.py
uv run pytest -q tests/runtime/test_m2_agent_loop.py
```

然后运行完整检查：

```bash
uv lock --check
uv run python -m compileall -q forge tests
uv run pytest -q
git diff --check
```

如果完整测试失败：

1. 判断失败是否由本次修改引起。
2. 修复本次引入的回归。
3. 不得删除原测试或减弱断言。
4. 不得把未执行的测试报告为通过。
5. 对确实无关的既存失败给出命令、错误和证据。

## 最终输出要求

完成后请报告：

1. 精确根因。
2. 修改文件列表。
3. Verification Ledger 原先在哪一步丢失产物信息。
4. 新的可信验证产物判定规则。
5. 如何防止验证后篡改产物被错误信任。
6. 为什么生成物仍不能充当任务进展。
7. 为什么未声明输出和验证副作用仍然被拒绝。
8. 新增和修改的测试。
9. 实际运行的每条验证命令及结果。
10. `git diff --stat`。
11. 尚存的边界情况。

请现在直接检查仓库、复现失败、修改源码、运行测试并完成修复。不要停留在方案阶段。
