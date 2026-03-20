---
name: terminal-orchestrate
description: |
  终端编排：本地 Agent 作为 orchestrator，通过 Zellij 或 tmux 控制 worker pane 执行任务。
  当用户需要在远程机器、容器、或本地另一个终端中执行命令和任务时使用此 skill。
  触发场景包括：提到 "远程执行"、"tmux 编排"、"zellij 编排"、"跨机器"、"远程容器"，
  或需要把任务委托给另一个 pane / 终端 / 机器执行时。即使用户没有明确说 "terminal-orchestrate"，
  只要涉及 "帮我在服务器上跑"、"让远程机器执行"、"SSH 过去跑一下" 等场景，都应触发。
---

# 终端编排 (Terminal Orchestrate)

你是 orchestrator，通过 **Zellij**（优先）或 **tmux** 控制一个或多个 worker pane 完成任务。
Worker 可以是远程 SSH 机器上的 Agent 实例，也可以是普通 shell。

## 第零步：探测后端

在做任何事之前，先探测当前环境，确定使用 Zellij 还是 tmux：

```bash
# 检查是否在 Zellij 内
echo $ZELLIJ
echo $ZELLIJ_SESSION_NAME

# 检查是否在 tmux 内
echo $TMUX
```

**决策优先级：**
1. 如果 `$ZELLIJ` 非空 → 使用 Zellij 模式
2. 如果 `$TMUX` 非空 → 使用 tmux 模式
3. 都不在 → 优先尝试启动 Zellij，回退到 tmux

---

## Zellij 模式

### 启动编排 session

```bash
# 双 worker 布局
zellij -s work --layout ~/.config/zellij/layouts/orchestrate.kdl

# 单 worker 布局
zellij -s work --layout ~/.config/zellij/layouts/orchestrate-single.kdl
```

### 核心操作：通过 orchestrator 插件（推荐）

如果 WASM 插件已编译（`~/.config/zellij/plugins/orchestrator.wasm` 存在），使用 pipe 协议：

| 操作 | 命令 |
|------|------|
| 发指令 | `echo '{"action":"send","target":"worker-1","command":"echo hello"}' \| zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate` |
| 列出 pane | `echo '{"action":"list-panes"}' \| zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate` |
| 聚焦 pane | `echo '{"action":"focus","target":"worker-1"}' \| zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate` |

#### 插件通信协议

发送 JSON 到 pipe：

```json
// 向指定 worker 发命令
{"action": "send", "target": "worker-1", "command": "echo hello"}

// 查询所有 pane 状态
{"action": "list-panes"}

// 聚焦指定 pane
{"action": "focus", "target": "worker-1"}
```

### 核心操作：CLI 简化模式（无插件，2-pane）

如果插件不可用，用 Zellij CLI 直接操作（仅适用于 2-pane 布局）：

| 操作 | 命令 |
|------|------|
| 发指令 | `zellij action move-focus right && zellij action write-chars "命令" && zellij action write 10 && zellij action move-focus left` |
| 读输出 | `zellij action dump-screen /tmp/worker-out.txt && cat /tmp/worker-out.txt \| tail -100` |
| 判断空闲 | 读 dump 文件末尾匹配提示符 |

**注意：** `write 10` 发送 Enter 键（ASCII 10 = LF）。

### 连接远程环境（按需）

通过插件或 CLI 模式向 worker 发送连接命令：

```bash
# SSH 直连
echo '{"action":"send","target":"worker-1","command":"ssh devserver"}' | \
  zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate

# 验证连接
sleep 3
echo '{"action":"send","target":"worker-1","command":"echo CONNECTED && hostname"}' | \
  zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate
```

### 读取 Worker 输出

Zellij 的 dump-screen 需要先切焦点到目标 pane：

```bash
# 方法 1：通过插件聚焦后 dump
echo '{"action":"focus","target":"worker-1"}' | \
  zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate
zellij action dump-screen /tmp/worker-out.txt
cat /tmp/worker-out.txt | tail -100

# 方法 2：让 worker 写文件
echo '{"action":"send","target":"worker-1","command":"cmd > /tmp/result.txt 2>&1"}' | \
  zellij pipe --plugin file:~/.config/zellij/plugins/orchestrator.wasm --name orchestrate
# 等完成后直接读文件（如果是本地 pane）
cat /tmp/result.txt
```

### 判断 Worker 是否空闲

```bash
wait_worker_zellij() {
  local target="${1:-worker-1}"
  local max_wait="${2:-300}"
  local elapsed=0
  local plugin="file:$HOME/.config/zellij/plugins/orchestrator.wasm"
  while [ $elapsed -lt $max_wait ]; do
    # Focus target and dump screen
    echo "{\"action\":\"focus\",\"target\":\"$target\"}" | zellij pipe --plugin "$plugin" --name orchestrate
    zellij action dump-screen /tmp/worker-screen.txt
    local tail=$(tail -5 /tmp/worker-screen.txt)
    # Agent idle detection
    if echo "$tail" | grep -q "? for shortcuts"; then
      echo "IDLE (agent)"
      return 0
    fi
    # Shell idle detection
    if echo "$tail" | grep -qE '[\$%#] $'; then
      echo "IDLE (shell)"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "TIMEOUT after ${max_wait}s"
  return 1
}
```

---

## tmux 模式

### 核心操作

| 操作 | 命令 |
|------|------|
| 发指令 | `tmux send-keys -t <target> "命令" Enter` |
| 读输出 | `tmux capture-pane -t <target> -p -S -100` |
| 判断空闲 | 读最后几行，匹配提示符 |

### 初始化 tmux 环境

```bash
# 检查已有 session
tmux list-sessions 2>/dev/null
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_width}x#{pane_height}' 2>/dev/null

# 创建 session + worker pane
tmux new-session -d -s work -x 200 -y 50 2>/dev/null || true
WINDOW=$(tmux list-windows -t work -F '#{window_index}' | head -1)
tmux split-window -v -t "work:${WINDOW}"
tmux list-panes -t "work:${WINDOW}" -F '#{pane_index} #{pane_id}'
```

**重要**：pane 编号不要硬编码，一定通过 `list-panes` 探测实际编号。

### 连接远程环境（按需）

```bash
# SSH 直连
tmux send-keys -t $WORKER 'ssh devserver' Enter
sleep 3

# SSH + Docker 容器
tmux send-keys -t $WORKER 'ssh devserver' Enter
sleep 3
tmux send-keys -t $WORKER 'docker exec -it mycontainer bash' Enter
sleep 2
```

### 判断 Worker 是否空闲

```bash
wait_worker_tmux() {
  local target="${1:-$WORKER}"
  local max_wait="${2:-300}"
  local elapsed=0
  while [ $elapsed -lt $max_wait ]; do
    local tail=$(tmux capture-pane -t "$target" -p | tail -5)
    if echo "$tail" | grep -q "? for shortcuts"; then
      echo "IDLE (agent)"
      return 0
    fi
    if echo "$tail" | grep -qE '[\$%#] $'; then
      echo "IDLE (shell)"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "TIMEOUT after ${max_wait}s"
  return 1
}
```

### 多 Worker 并行

```bash
tmux split-window -h -t $WORKER
tmux list-panes -t work -F '#{pane_index} #{pane_id} #{pane_current_command}'
WORKER_A="work:${WINDOW}.2"
WORKER_B="work:${WINDOW}.3"  # 实际编号以 list-panes 为准
```

---

## 工作规范

### 作为 orchestrator 你应该：

1. **收到任务后先拆解**，想清楚哪些子任务发给 worker，哪些自己做
2. **逐步发送，等每步完成再发下一步**，除非任务间无依赖可以并行
3. **每次都读输出确认结果**，不要盲发
4. **结果有问题就发修正指令给 worker**，不要自己在本地改远程文件
5. **向用户汇报进度**，尤其是 worker 返回了关键结果时

### 注意事项

- **pane 编号/名称**：Zellij 用名称（layout 中定义），tmux 用编号（需探测）
- **Zellij 插件权限**：首次加载时需要用户授权，之后缓存
- **tmux history**：默认滚动缓冲有限，必要时 `tmux set-option -t work history-limit 50000`
- **SSH 断连恢复**：建议在远程也用 tmux/zellij，断开后进程仍存活
- **API 限流**：如果 orchestrator 和 worker 都跑 Agent 且共享配额，可能互相 429
- **特殊字符**：send-keys/write-chars 中的 `"` `'` `$` 等需要正确转义

## 故障排查

| 问题 | 解决 |
|------|------|
| Zellij pipe 无响应 | 检查插件是否加载：`echo '{"action":"list-panes"}' \| zellij pipe ...` |
| tmux send-keys 无效果 | 检查 target：`tmux list-panes -t work` |
| 输出读不到 | Zellij: dump-screen 需先聚焦目标 pane；tmux: 增大 `-S` 行数 |
| worker 无响应 | 可能在执行长任务，读输出末尾确认 |
| SSH 断开 | 在远程用 tmux/zellij 保活 |
| pane 编号对不上 | 永远用 list-panes/list 探测 |

## 清理

```bash
# Zellij
zellij kill-session work

# tmux
tmux kill-session -t work
```
