# Hello-Agents - Simple Travel Assistant 示例

这是 [Datawhale Hello-Agents](https://github.com/datawhalechina/Hello-Agents)
教程学习过程中的一个小型练习项目，用于演示如何构建一个能够调用工具的旅行助手 Agent。

本仓库是独立示例，不是 `datawhalechina/Hello-Agents` 官方仓库。

## 功能

- 使用 `wttr.in` 查询城市天气
- 使用 Tavily 根据天气推荐旅游景点
- 使用 DeepSeek 兼容接口驱动 Agent
- 通过简单的 Thought / Action / Observation 循环完成任务

## 运行

先设置 API Key：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:TAVILY_API_KEY="你的 Tavily API Key"
```

然后运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

这是一个用于学习 Hello-Agents 教程中 Agent 基本工作流程的最小示例。

## 简单的 Memory Agent

`memory_reflection_agent.py` 在基础 ReAct 流程上增加了一个简单的本地记忆模块：

- 从用户输入中提取兴趣、预算和避开项，并保存到 `agent_memory.json`
- 记录用户接受或拒绝推荐的反馈，帮助下一轮调整结果
- 连续拒绝 3 次后触发 Reflection，更新推荐策略并生成澄清问题
- 发现景点售罄时，自动根据当前偏好查询备选景点

运行 Memory Agent：

```powershell
.\.venv\Scripts\python.exe memory_reflection_agent.py
```

`agent_memory.json` 只保存在本地，已加入 `.gitignore`，不会被提交到远端。
