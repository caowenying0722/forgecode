请基于 ForgeCode 当前最新代码完成一次纵向重构。本轮重点解决“完成判定针对特定游戏任务硬编码、ProgressEvaluator 已有能力未接入主循环、计划更新被错误当作有效进展”的问题。

开始工作前，请先阅读当前实际代码和测试，不要根据旧提示词重新实现已经存在的功能。当前仓库已经具备以下能力，应保留并复用：

* TaskContract 和多种 TurnKind。
* 确定性意图快速路径。
* 生产可用的 ModelSemanticTaskClassifier。
* policy_requires_change 不再覆盖咨询和只读回合。
* TurnRuntimeState。
* VerificationLedger。
* filesystem_revision/source_revision。
* 验证生成物分类。
* verify 与正式验证命令的统一记录。
* RepairTarget。
* 验证恢复的动态读取预算。
* Session rollout、会话恢复和多模型能力。

除非发现明确 Bug，不要重新设计或平行实现上述模块。

请重点审查：

* forge/runtime/completion_checker.py
* forge/runtime/progress.py
* forge/runtime/agent_loop.py
* forge/runtime/task_model.py
* forge/runtime/agent_controller.py
* forge/runtime/request_builder.py
* forge/runtime/recovery_manager.py
* forge/runtime/verification.py
* task_plan、task_update、finish_task 对应工具
* 相关单元测试和轨迹测试

先输出一份简短审计，确认以下观察是否成立：

1. completion_checker.py 中存在针对“三选一升级、武器组合、Boss”的领域特定验收正则。
2. finish_task 主要通过扫描变更文件和证据文本来推断功能是否完成。
3. ProgressEvaluator 已支持 completed_acceptance_criteria、completed_plan_step、验证错误数量和 failure_signature_changed。
4. agent_loop.py 调用 ProgressEvaluator 时没有完整传入上述信号。
5. 成功执行 task_update 会直接把 batch.task_progressed 设为 true，即使没有附带有效实现证据。
6. RequestState 仍保留部分旧兼容字段，但生产路径已经优先使用 TurnRuntimeState。
7. Action Recovery 和 Mutation Recovery 仍存在 read_available 布尔式读取限制。

如果观察与最新代码不符，以真实调用链为准，并在实施前说明差异。

本轮只实施“通用验收证据与进展判断”这个里程碑。不要同时大规模重写 Recovery 或 agent_loop.py。

第一部分：删除领域硬编码验收。

删除或废弃 `_feature_acceptance_criteria()` 以及所有专门识别升级、武器组合、Boss、PlayScene、Phaser 等业务词汇的完成判断。

Completion Checker 的核心逻辑不得包含具体产品功能名称、框架名称或测试 Fixture 名称。

不要用更多关键词正则替代当前正则。

TaskContract、task_plan 或 SemanticTaskClassifier 应在任务开始时产生具体的 deliverables 和 acceptance criteria。Completion 只消费这些结构化条件，不负责重新猜测用户想实现什么。

第二部分：建立通用 Acceptance Evidence 模型。

实现 `AcceptanceEvidence`、`AcceptanceLedger` 或等价结构，用于记录每一条验收条件是否已经具备证据。

建议至少包含：

* criterion_id
* criterion_text
* status：pending、partially_satisfied、satisfied、blocked
* evidence_type
* evidence_paths
* symbols
* plan_step_id
* verification_record_ids
* source_revision
* producer：tool、model、runtime、test
* confidence
* explanation

证据类型至少支持：

* source_change
* test_result
* typecheck
* build
* lint
* smoke
* symbol_evidence
* runtime_integration
* configuration
* review
* manual_limitation

验收条件不能仅因某个关键词出现在代码中就自动满足。

例如“运行时可以选择一个升级”不能仅凭存在 `upgrade` 字符串满足。它至少需要相关代码接入证据，并根据 VerificationPolicy 要求测试、smoke 或明确标记为未进行运行时验证。

对于无法自动验证的行为，应保留为 partially_satisfied 或 manual_limitation，不能伪装成已完成。

第三部分：让计划步骤与验收条件建立关联。

task_plan 中的步骤应能够声明它负责哪些 deliverable 和 criterion_id。

task_update 将步骤标记为 completed 时，必须附带结构化 evidence，例如：

* 修改过的目标文件。
* 相关符号。
* 对应的 VerificationRecord。
* 解决的 RepairTarget。
* 满足的 criterion_id。

仅把 Todo 或计划步骤状态改成 completed，不能自动算有效进展。

如果 task_update 没有证据，可以更新展示状态，但不得重置停滞计数，也不得增加完成度。

第四部分：把 ProgressEvaluator 已有参数真正接入主循环。

修改 agent_loop.py 的事件收集或 batch 结构，让 ProgressEvaluator 实际接收到：

* completed_acceptance_criteria
* completed_plan_step
* previous_verification_error_count
* current_verification_error_count
* failure_signature_changed
* verification_reused
* 当前 source_revision 的相关源码变化
* 当前 RepairTarget 是否被解决

删除“只要 task_update 成功就设置 task_progressed=True”的宽松逻辑。

建议把 task_progressed 改成结构化结果，而不是单一布尔值，例如：

* completed_step_ids
* completed_criterion_ids
* attached_evidence
* evidence_valid

以下行为不能算有效进展：

* 重复调用 todo_write。
* 只修改计划文字。
* 无证据地把步骤标记完成。
* 重复读取。
* 缓存命中。
* 只生成构建产物。
* 创建临时文件。
* 原样重复验证。
* 修改与当前 priority、计划步骤或 RepairTarget 无关的文件。
* 子 Agent 达到轮次上限但没有返回报告。

以下行为可以算有效进展：

* 满足一条有证据的 acceptance criterion。
* 完成一个有证据的计划步骤。
* 产生当前目标范围内的有效源码、测试或配置变化。
* 验证错误数量下降。
* 失败签名变化。
* RepairTarget 被解决。
* Verification 从 FAILED 变为 PASSED。
* 获取当前步骤此前缺失且必要的新证据。

第五部分：重构 finish_task 的差距报告。

finish_task 不再扫描业务关键词推断完成度，而是读取：

* TaskContract.deliverables
* TaskContract.acceptance_criteria
* 当前计划步骤
* AcceptanceLedger
* VerificationLedger
* WorkspaceChangeClassifier
* RepairTarget

完成声明被拒绝时，返回结构化差距：

* completed_deliverables
* pending_deliverables
* satisfied_criteria
* partially_satisfied_criteria
* missing_criteria
* missing_evidence
* pending_plan_steps
* missing_verification
* unresolved_repair_target
* unrelated_changes
* forbidden_changes
* recommended_next_action

不要只返回笼统的 “declaration did not match available execution evidence”。

构建通过只能满足 build 条件，不能自动证明业务功能已经实现。

第六部分：让规则具备跨项目通用性。

请使用至少三类完全不同的任务证明新模型不是为当前游戏项目定制：

场景 A：TypeScript 游戏功能。

目标是新增一个运行时升级机制。需要源码接入、类型检查和 smoke evidence。不能通过搜索 upgrade 关键词自动完成。

场景 B：Python Bug 修复。

目标是修复除零行为并补测试。验收条件应由代码变化和 pytest 结果满足，不应要求任何游戏或运行时场景概念。

场景 C：文档或只读任务。

目标是分析架构或修改文档。不得要求业务功能 smoke test，也不能因为没有源码变化而失败。

核心 Completion、Progress 和 Acceptance 代码中不得出现这些示例的具体功能名称。

第七部分：增加轨迹级回归测试。

至少增加以下测试：

1. “三选一升级 + 武器组合 + Boss”任务不再触发领域硬编码正则。
2. TaskContract 的验收条件可以由通用 AcceptanceLedger 跟踪。
3. 仅在源码中出现 “upgrade” 或 “boss” 字符串不能满足验收条件。
4. task_update 没有 evidence 时不算进展。
5. task_update 附带有效源码和验证证据时可以完成对应步骤。
6. Todo 文本更新不重置停滞计数。
7. 验证错误数量下降算进展。
8. 相同失败签名、无相关修改、原样再次验证不算进展。
9. RepairTarget 被相关修改解决后算进展。
10. build passed 只能满足 build criterion。
11. 业务行为缺少 test/smoke 时保持 partially_satisfied。
12. finish_task 返回具体缺失 criterion，而不是通用 evidence mismatch。
13. Python Bug Fixture 可以使用同一套验收证据机制完成。
14. advisory 和 inspect 回合不创建无意义的代码验收条件。
15. 回放同一事件轨迹能够重建相同的 AcceptanceLedger 状态。

第八部分：保留最新功能。

本轮不得破坏：

* 多模型选择和模型配置。
* Session rollout。
* resume、fork 和历史恢复。
* source_revision/filesystem_revision。
* VerificationLedger。
* 验证产物分类。
* MCP。
* 权限系统。
* Subagent 和 Team API。
* CLI 现有命令。

AcceptanceEvidence、计划步骤完成和 finish_task 结果应进入追加式 Session Rollout，使会话恢复后仍能重建验收状态。

实施边界：

* 不要新增平行状态机。
* 不要新增 recovery 布尔变量。
* 不要继续增加意图关键词正则。
* 不要大规模重写 agent_loop.py。
* 不要针对 Phaser、PlayScene、Boss、升级或当前测试仓库硬编码。
* 不要删除已经完成的 VerificationLedger 和 revision 分类机制。
* 所有新行为必须通过单元测试和轨迹级测试证明。
* 优先以小型数据结构和事件接线完成纵向切片。

建议拆成三个提交：

提交一：

* 增加通用 AcceptanceEvidence/AcceptanceLedger。
* 删除领域特定 `_feature_acceptance_criteria()`。
* 增加独立单元测试。

提交二：

* 将 task_plan、task_update、VerificationLedger 和 AcceptanceLedger 接通。
* 将 ProgressEvaluator 的高级进展参数接入 agent_loop。
* 删除无证据 task_update 自动算进展的逻辑。

提交三：

* 重构 finish_task 差距报告。
* 增加跨项目轨迹测试。
* 将验收证据写入 Session Rollout。
* 更新 README 和 FORGECODE_HANDOFF.md。

本轮暂时不要完整迁移 RequestState 兼容字段，也不要完整重写 Action Recovery。请将以下内容整理到 PLANS.md，作为下一里程碑：

* 移除 RequestState 中的标量兼容恢复字段。
* 将 agent_loop.py 的剩余局部计数器迁移进 TurnRuntimeState。
* 使用统一 RecoveryScope 替代 action/mutation recovery 的 read_available 布尔值。
* 为编辑操作增加 expected file revision/hash。
* 进一步拆分 3100 行的 agent_loop.py。

完成后运行：

* `uv lock --check`
* `uv run python -m compileall -q forge tests`
* `uv run pytest -q`
* `git diff --check`

最终报告必须包含：

* 删除了哪些领域硬编码。
* AcceptanceEvidence 的数据模型。
* 验收条件如何获得和失去证据。
* task_update 为什么不再自动算进展。
* ProgressEvaluator 哪些原有参数已经真正接入。
* finish_task 的新差距报告示例。
* 新增的跨项目测试。
* Session Rollout 如何保存验收证据。
* 测试总数及结果。
* 下一里程碑仍需迁移的兼容字段和恢复逻辑。
