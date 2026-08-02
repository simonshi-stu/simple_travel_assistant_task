import os
import re

from attraction_tool import get_attraction
from deepseek_client import DeepSeekCompatibleClient
from weather_tool import get_weather


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


AGENT_SYSTEM_PROMPT = """你是一个旅行助手，可以使用以下工具：
1. get_weather(city="城市名")：查询指定城市的实时天气。
2. get_attraction(city="城市名", weather="天气信息")：根据城市和天气推荐景点。

你每次只能执行一个行动，并且必须严格使用以下格式回复：
Thought: 描述你的思考
Action: 工具名(参数名="参数值")

获得 Observation 后继续思考并执行下一步。当任务完成时，使用：
Thought: 描述你的最终思考
Action: Finish[最终答案]
"""


API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_ID = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-v4-flash")

llm = DeepSeekCompatibleClient(
    api_key=API_KEY,
    base_url=BASE_URL,
)


user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "=" * 40)


for i in range(5):
    print(f"--- 循环 {i + 1} ---\n")

    full_prompt = "\n".join(prompt_history)

    response = llm.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ],
    )
    llm_output = response.choices[0].message.content or "" #取第一个候选答案里的消息正文

    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        llm_output,
        re.DOTALL,
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")

    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = (
            "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 "
            "'Thought: ... Action: ...' 的格式。"
        )
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str, re.DOTALL).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
