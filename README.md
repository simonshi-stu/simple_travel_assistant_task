# Hello Agent - Simple Travel Assistant

这是 [Hello Agent](https://github.com/simonshi-stu/simple_travel_assistant_task) 项目下的一个小型示例，用于演示一个可以调用工具的旅行助手 Agent。

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

这是一个用于学习 Agent 基本工作流程的最小示例。
