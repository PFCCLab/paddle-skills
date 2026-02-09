# Paddle Skills

面向 PaddlePaddle 相关项目的 AI 编码助手 skills 集合。

## 可用 Skills

### Paddle 核心框架

| Skill          | 说明                                                          |
| -------------- | ------------------------------------------------------------- |
| `paddle-pull-request` | 创建或更新 PR，强制使用 Paddle 官方 PR 模板补全 PR 描述内容。 |
| `paddle-debug` | 调试 Paddle 相关问题，生成调试报告。 |

### FastDeploy

| Skill          | 说明                                                          |
| -------------- | ------------------------------------------------------------- |
| `fastdeploy-pull-request` | 创建或更新 PR，遵循 FastDeploy 官方 PR 模板和 CI 检查规范。 |

## 安装

使用 `bunx` 安装本仓库中的指定 skill，例如 `paddle-pull-request`：

```bash
bunx skills add PFCCLab/paddle-skills --skill "paddle-pull-request"
```

安装成功后，AI 助手即可在需要创建或更新 Paddle 相关仓库的 PR 时调用 `paddle-pull-request` skill，并按照 Paddle 官方模板生成或补全 PR 描述。

# 参考项目

- https://github.com/junoh-moon/skills
