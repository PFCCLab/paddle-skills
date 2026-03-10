---
name: paddle-ci
description: |
  分析 Paddle 相关仓库 PR 的 CI 失败原因，适用于 checks 失败、Approval 卡住、PR 模板/代码风格校验异常、单测失败或 flaky 问题排查。
  先判断是否为 PR 引入的问题，再对比其他 PR / develop 状态，输出诊断报告与修复计划，并在合适时机使用 /re-run all-failed 或推动 reviewer 处理。
---

# Paddle CI 失败分析与修复

## 核心原则

**先诊断，再修复** — 不要在没有充分分析的情况下盲目修改代码。

## 流程

### 1. 快速分诊

| 现象 | 常见归属 | 先看哪里 | 常见动作 |
|-----|---------|---------|---------|
| `CheckPRTemplate` 失败 | PR 内容问题 | PR 描述、模板检查日志 | 补全模板字段 |
| `Codestyle-Check` 失败 | PR 引入的问题 | pre-commit / clang-format / cpplint 日志 | 本地执行 pre-commit 修复 |
| `Approval` 失败 | 流程问题 | 日志中的 required approvers | 在 PR 中 @ 对应 reviewer |
| 主 CI / 单测失败 | 代码或环境问题 | 失败 job、失败用例、develop 同类结果 | 先判断是否为 PR 引入 |
| 同一测试偶发失败 | Flaky / 基础设施问题 | 最近 PR、develop、重复运行结果 | 记录证据后再决定是否重试 |

### 2. 判断是否为 PR 引入的问题

先确认失败属于哪一类：**PR 引入 / develop 已有问题 / 基础设施问题 / flaky**。

```bash
# 查看当前 PR 的失败 job
gh pr checks <PR_NUMBER>

# 查看 develop 最近运行结果，判断是否已有同类失败
gh run list --branch develop --limit 5 \
  --json databaseId,workflowName,displayTitle,conclusion,createdAt \
  | jq '.[] | {id: .databaseId, workflow: .workflowName, title: .displayTitle, conclusion, createdAt}'

# 查看失败 run 的 job / step 信息
gh run view <RUN_ID> --json jobs \
  | jq '.jobs[] | {name, conclusion, failed_steps: [.steps[]? | select(.conclusion == "failure") | .name]}'

# 仅在需要定位关键信息时查看失败日志
gh run view <RUN_ID> --log-failed
```

**判断依据**：

- develop 正常、当前 PR 失败，且错误与改动相关 → **PR 引入的问题**
- develop 也失败，且错误一致 → **大概率不是当前 PR 引入**
- 多个 PR 同时出现同类报错 → **优先怀疑基础设施或已有问题**
- 同一用例偶发失败、重跑结果不稳定 → **Flaky test**

### 3. 检查其他 PR 是否有相似问题

如果无法直接判断归属，再看最近其他 PR 是否出现相同 job / 相同错误关键词。

```bash
# 查看最近 PR
gh pr list --state all --limit 10 --json number,title,headRefName,url \
  | jq '.[] | {number, title, branch: .headRefName, url}'

# 查看相似 PR 的 checks
gh pr checks <PR_NUMBER>
```

如果多个 PR 都有相同问题，优先考虑：

- 基础设施问题（CI 环境、网络等）
- develop 分支已有的问题
- Flaky test

### 4. 生成分析报告和修复计划

先给出归因和修复计划，再动手修改。

**最简报告模板**：

```markdown
### 归属
[PR 引入 / develop 已有 / 基础设施 / flaky]

### 证据
[失败 job、关键报错、develop / 其他 PR 对比结论]

### 修复计划
[准备修改什么；如果不改代码，说明将 @ 谁、是否 /re-run all-failed、如何验证]
```

### 5. 实际修复与验证

- **PR 引入的问题**：本地复现后修复，再推送验证。
- **Approval / 模板类问题**：直接补内容或 @ 对应 reviewer，不要改无关代码。
- **基础设施 / flaky**：先保留证据；确认不是代码问题后，再考虑 `/re-run all-failed`。
- **develop 已有问题**：明确标注不是当前 PR 引入，避免误修。

## Paddle CI 检查项

### PR 模板检查 (CheckPRTemplate)

检查 PR 描述是否符合 Paddle 官方模板要求。

**失败处理**：

- 按照 [PR 模板](https://github.com/PaddlePaddle/Paddle/blob/develop/.github/PULL_REQUEST_TEMPLATE.md) 补充完整 PR 描述
- 参考 [PR 模板说明](https://github.com/PaddlePaddle/Paddle/wiki/PULL-REQUEST-TEMPLATE--REFERENCE)
- 优先修正文案和缺失字段，不要把模板失败误判成代码问题

### 代码风格检查 (Codestyle-Check)

运行 pre-commit 检查代码风格，包括 clang-format、cpplint、ast-grep 等。

**本地修复**：

```bash
# 安装依赖
pip install pre-commit==2.17.0 cpplint==1.6.0 clang-format==13.0.0

# 运行检查并自动修复
pre-commit run --files $(git diff --name-only origin/develop)

# 或检查所有文件
pre-commit run --all-files
```

优先关注格式化、lint、规则扫描失败；这类问题通常不需要分析通用 CI 基础设施。

### Approval 检查

某些目录的修改需要特定 reviewer 的 approve。

**处理方式**：

- 查看 CI 日志中提示的 required approvers
- 在 PR 中 @ 对应的 reviewer 请求 review / approve
- 如果只是缺少 approval，不要继续修改代码来“碰运气过 CI”

### 主 CI 流水线

| Job | 重点关注 |
|-----|---------|
| **Linux-CPU** | 编译错误、常规单测失败，优先判断是否与本次改动直接相关 |
| **Linux-XPU** | 昆仑芯相关适配、设备分支逻辑、特定测试用例 |
| **Linux-DCU** | 海光 DCU 相关兼容性、条件分支或特定算子实现 |
| **Linux-NPU** | 昇腾 NPU 相关路径、设备能力差异、用例兼容性 |
| **Mac-CPU** | macOS 平台差异、编译器/系统行为差异 |
| **PR-CI-SOT** | SOT (Symbolic OpTest) 相关行为变化 |
| **Distribute-stable** | 分布式训练、集群/通信相关用例 |

先判断失败 job 是否落在本次改动涉及的能力范围内，再决定是否需要修代码。

### 单元测试失败

优先提取 **失败用例名 + 首个有效报错**，再做本地复现。

```bash
# 本地复现（需要编译 Paddle）
ctest -R test_name -V

# 或使用 pytest
pytest path/to/test_file.py::test_name -v --tb=long
```

如果是设备相关 job 失败，优先检查该设备路径是否被本次改动影响。

### Flaky Test

只有在已有证据表明问题不是稳定复现的代码缺陷时，才按 flaky 处理。

多次运行确认稳定性：

```bash
for i in {1..10}; do
  ctest -R test_name -V || echo "Failed on run $i"
done
```

确认属于 flaky 或基础设施波动后，可使用 `/re-run all-failed` 作为重试手段；不要把它当成第一反应。

## 禁止事项

- **禁止通过降低质量门槛来通过 CI**
  - ❌ 直接跳过测试、注释断言、放宽检查条件
  - ✅ 找到失败根因后修复，或明确说明是非 PR 引入的问题
- **禁止在没有分析的情况下盲目重试**
  - ❌ 一看到红灯就先 `/re-run all-failed`
  - ✅ 先判断是否为 flaky / 基础设施问题，再决定是否重试
- **禁止把流程问题误当成代码问题**
  - ❌ Approval 没过却反复提交无关代码
  - ✅ 查看 required approvers，并在 PR 中 @ 相关 reviewer
- **禁止忽略 develop / 其他 PR 的对比结果**
  - ❌ develop 也同样失败，还直接认定是当前 PR 引入
  - ✅ 先做横向对比，再决定修复方向

## 参考文档

- [Paddle CI 手册](https://github.com/PaddlePaddle/Paddle/wiki/paddle_ci_manual)
- [PR 模板说明](https://github.com/PaddlePaddle/Paddle/wiki/PULL-REQUEST-TEMPLATE--REFERENCE)

## 完成后

更新分析结论与修复计划；如果已经完成修复并验证通过，再使用 `paddle-pull-request` skill 创建或更新 PR。
