请阅读 `README.md` 和 `CONTRIBUTING.md`，了解仓库说明与贡献规范。

## PR 标题规范

PR 标题必须遵循以下格式（详见 `CONTRIBUTING.md`）：

```
<emoji> <type>(<scope>): <subject>
```

- `✨ add(<skill-name>)` — 新增 skill，**scope 必填**（填 skill 名）
- `⬆️ update(<skill-name>)` — 修改/增强 skill，**scope 必填**
- `🐛 fix(<skill-name>)` — 修复 skill 问题，**scope 必填**
- `📝 docs` — 文档调整，scope 可选
- `🔧 chore` — CI、基础设施等杂项，scope 可选

> `✨ add` / `⬆️ update` / `🐛 fix` 仅用于 skill 的变更；CI 或基础设施改动请使用 `🔧 chore`。

示例：

- `✨ add(paddle-trace): initial version`
- `⬆️ update(paddle-pull-request): support multi template`
- `🐛 fix(paddle-debug): handle empty log path`
- `📝 docs: update contributing.md`
- `🔧 chore: add CI workflow for PR title validation`
