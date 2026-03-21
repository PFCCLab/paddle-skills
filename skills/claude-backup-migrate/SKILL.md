---
name: claude-backup-migrate
description: "备份、迁移、恢复 Claude Code 的对话历史和配置。当用户提到备份、迁移、恢复对话、换机器、导出 session 等需求时使用此 Skill。"
---

# Claude Code 备份迁移指南

帮助用户备份、迁移、恢复 Claude Code 的对话历史和配置数据。

## 平台差异速查

| 平台 | 配置目录 | Home 路径 | 项目目录名示例 |
|------|---------|----------|--------------|
| macOS | `~/.claude/` | `/Users/<username>/` | `-Users-me-projects-app` |
| Linux | `~/.claude/` | `/home/<username>/` | `-home-me-projects-app` |
| WSL | `~/.claude/` | `/home/<username>/` | `-home-me-projects-app` |
| Windows | `%USERPROFILE%\.claude\` | `C:\Users\<username>\` | `-C-Users-me-projects-app` |

以下命令以 macOS 为主，Linux 差异用 `# Linux:` 标注。

## 存储结构

```
~/.claude/
├── projects/                          # 所有对话 session（核心数据）
│   └── -<home-path-encoded>/          # 目录名 = 工作目录绝对路径中 / 替换为 -
│       ├── <uuid>.jsonl               # 每个对话一个 JSONL 文件
│       ├── <uuid>/
│       │   └── subagents/             # 子 agent 对话记录
│       └── memory/                    # 项目级持久记忆
├── settings.json                      # 全局配置（hooks、模型等）
├── history.jsonl                      # 命令输入历史
├── file-history/                      # 文件修改历史（按 session 分目录）
├── session-env/                       # session 环境变量快照
├── hooks/                             # Hook 脚本（如 peon-ping）
├── skills/                            # 已安装的 Skills
├── plugins/                           # 已安装的 Plugins
├── plans/                             # Plan mode 文件
├── debug/                             # 调试日志（可删除）
├── cache/                             # 缓存（可删除）
└── paste-cache/                       # 粘贴板缓存（可删除）
```

## 文件重要性

| 文件/目录 | 迁移优先级 | 说明 |
|----------|----------|------|
| `projects/` | 必须 | 所有对话内容和项目记忆 |
| `settings.json` | 必须 | hooks 配置、模型设置 |
| `hooks/` | 推荐 | Hook 脚本（音效、自动化等） |
| `history.jsonl` | 推荐 | 命令输入历史 |
| `file-history/` | 可选 | 文件修改可恢复（体积可能较大） |
| `skills/` | 可选 | 可通过 `npx skills` 重新安装 |
| `plugins/` | 可选 | 可通过 marketplace 重新安装 |
| `session-env/` | 可选 | session 环境快照 |
| `debug/` | 不需要 | 调试日志，可删除节省空间 |
| `cache/` | 不需要 | 缓存，自动重建 |
| `paste-cache/` | 不需要 | 粘贴板缓存 |

## 一、备份

根据用户需求选择合适的备份范围：

```bash
# 完整备份（推荐）
tar czf claude-backup-$(date +%Y%m%d).tar.gz -C ~ .claude/

# 只备份对话 session
tar czf claude-sessions-$(date +%Y%m%d).tar.gz -C ~ .claude/projects/

# 只备份配置（不含对话）
tar czf claude-config-$(date +%Y%m%d).tar.gz \
  -C ~ .claude/settings.json .claude/hooks/ .claude/skills/ .claude/plugins/

# 精简备份（排除调试和缓存）
tar czf claude-backup-slim.tar.gz -C ~ \
  --exclude='.claude/debug' \
  --exclude='.claude/cache' \
  --exclude='.claude/paste-cache' \
  .claude/
```

## 二、迁移到新机器

### 步骤 1：打包

```bash
tar czf claude-full.tar.gz -C ~ .claude/
```

### 步骤 2：传输

```bash
scp claude-full.tar.gz <newmachine>:~/
```

### 步骤 3：解压

```bash
cd ~ && tar xzf claude-full.tar.gz
```

### 步骤 4：路径修正（关键步骤）

恢复对话必须保证路径一致。Session 的归属取决于两个地方：

1. **目录名**：`projects/` 下的目录名编码了工作目录的绝对路径，`/` 替换为 `-`
2. **文件内容**：`.jsonl` 文件内每条消息记录中包含 `cwd` 字段，记录了当时的工作目录

如果新机器的用户名或项目路径不同，两处都需要修改。

#### 同平台迁移（macOS -> macOS 或 Linux -> Linux）

只需替换用户名：

```bash
cd ~/.claude/projects/

OLD_USER="olduser"
NEW_USER="newuser"

# 第一步：重命名目录
for d in *${OLD_USER}*; do
  mv "$d" "$(echo "$d" | sed "s/${OLD_USER}/${NEW_USER}/g")"
done

# 第二步：修改 jsonl 文件内的 cwd 信息
# macOS:
find ~/.claude/projects/ -name "*.jsonl" -exec \
  sed -i '' "s|/Users/${OLD_USER}/|/Users/${NEW_USER}/|g" {} +

# Linux:
find ~/.claude/projects/ -name "*.jsonl" -exec \
  sed -i "s|/home/${OLD_USER}/|/home/${NEW_USER}/|g" {} +
```

#### 跨平台迁移（macOS -> Linux）

路径前缀也要改：

```bash
cd ~/.claude/projects/

OLD_PREFIX="-Users-olduser"
NEW_PREFIX="-home-newuser"

# 第一步：重命名目录
for d in ${OLD_PREFIX}-*; do
  mv "$d" "$(echo "$d" | sed "s|${OLD_PREFIX}|${NEW_PREFIX}|g")"
done

# 第二步：修改 jsonl 文件内的 cwd
# Linux:
find ~/.claude/projects/ -name "*.jsonl" -exec \
  sed -i 's|/Users/olduser/|/home/newuser/|g' {} +
```

#### 跨平台迁移（macOS -> Windows）

Windows 路径编码不同（`C:\` -> `-C-`），建议使用 WSL 作为中转：

```powershell
# Windows PowerShell 下操作
$old = "/Users/olduser/"
$new = "C:/Users/newuser/"
Get-ChildItem "$env:USERPROFILE\.claude\projects" -Recurse -Filter "*.jsonl" |
  ForEach-Object { (Get-Content $_.FullName) -replace [regex]::Escape($old), $new | Set-Content $_.FullName }
```

> 可以在对话中输入 `/status` 查看当前 Session ID 和工作目录等信息。

### 步骤 5：重装 Hook / Skill 依赖（如有必要）

Hook 脚本和 Skill 文件会一起迁移，但如果依赖了外部工具，需要在新机器重新安装。

### 步骤 6：恢复权限

```bash
chmod 700 ~/.claude/projects
find ~/.claude/projects -name "*.jsonl" -exec chmod 600 {} +
```

## 三、恢复对话

```bash
# 方式 1：交互式选择（推荐）
cd <项目目录>
claude            # 启动后输入 /resume，选择历史 session

# 方式 2：指定 Session ID（可通过 /status 命令查看）
claude --resume <session-uuid>

# 方式 3：继续最近一次对话
claude --continue
```

> 恢复条件：必须在对应的项目目录下启动 claude，才能看到该目录下的历史 session。如果路径不一致，需要先完成「步骤 4：路径修正」。

## 四、搜索历史对话

如果用户安装了 CASS，可以用它全文搜索：

```bash
cass search "关键词" --robot --limit 10
cass search "关键词" --robot --workspace /path/to/project
cass search "关键词" --robot --days 7
```

## 注意事项

1. **对话文件格式**：`.jsonl`（JSON Lines），每行一条消息记录
2. **项目路径映射**：目录名中 `/` 被替换为 `-`（macOS: `/Users/me/projects/app` -> `-Users-me-projects-app`）
3. **sed 命令差异**：macOS 的 `sed -i` 需要 `''` 参数，Linux 不需要
4. **大文件注意**：长对话的 session 文件可能超过 50MB，注意传输工具的大小限制
5. **hooks 路径**：`settings.json` 中的 hook command 使用绝对路径，跨机器需要检查并更新

## 执行策略

1. 先确认用户的操作系统和需求（备份/迁移/恢复）
2. 确认源机器和目标机器的平台和用户名
3. 根据场景选择对应的命令，**逐步执行并确认每一步结果**
4. 路径修正步骤务必让用户确认 OLD/NEW 变量值再执行
