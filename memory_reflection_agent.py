"""带记忆、备选推荐和反思机制的旅行助手示例。

这个文件是在原始 Thought-Action-Observation baseline 之上增加的教学版实现。
它刻意使用普通 Python 数据结构，而不是引入复杂的 Agent 框架，方便逐步阅读：

1. 用户输入先进入记忆提取模块；
2. LLM 根据记忆和本轮历史生成 Thought / Action；
3. Action 调用天气或景点工具；
4. Observation 被写回短期轨迹；
5. 发现门票售罄时自动查询备选；
6. 用户拒绝推荐时记录原因；
7. 连续拒绝达到 3 次时进入 Reflection 阶段，更新策略和记忆。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from attraction_tool import get_attraction
from deepseek_client import DeepSeekCompatibleClient
from weather_tool import get_weather


# ---------------------------------------------------------------------------
# 一、全局配置和工具注册
# ---------------------------------------------------------------------------

# 工具名和真正 Python 函数之间的映射。
# LLM 只能输出字符串形式的工具名，程序再通过这个字典找到函数并执行。
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

MODEL_ID = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-v4-flash")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MEMORY_FILE = Path(__file__).with_name("agent_memory.json")
MAX_REACT_STEPS = 6
REFLECTION_THRESHOLD = 3


# 这些词是一个非常简单的票务状态检测器。
# 真实项目中，最好增加一个专门的票务 API 工具，而不是只靠文本关键词判断。
SOLD_OUT_MARKERS = (
    "售罄",
    "售完",
    "没有票",
    "无票",
    "sold out",
    "fully booked",
    "tickets unavailable",
)


AGENT_SYSTEM_PROMPT = """你是一个带记忆的旅行助手。

你可以使用以下工具：
1. get_weather(city="城市名")：查询城市当前天气。
2. get_attraction(city="城市名", weather="天气和约束")：搜索景点推荐。

你必须严格按照下面格式输出，每次只执行一个 Action：
Thought: 说明当前思考
Action: 工具名(参数名="参数值")

如果已经有足够信息，使用：
Thought: 说明最终判断
Action: Finish[给用户的回答]

推荐时必须参考记忆中的兴趣、预算、避开项和反思策略。
如果观察结果中出现门票售罄，不要把售罄景点作为最终推荐，应继续调用景点工具寻找备选。
"""


REFLECTION_SYSTEM_PROMPT = """你是旅行助手的反思模块。
请分析用户连续拒绝推荐的原因，并输出严格合法的 JSON，不要输出 Markdown：
{
  "preference_updates": {
    "interests_add": [],
    "avoid_add": [],
    "budget": null
  },
  "new_strategy": "下一轮推荐策略",
  "clarifying_question": "如果仍然缺少关键信息，要主动询问的问题"
}
"""


# ---------------------------------------------------------------------------
# 二、记忆模块：长期记忆、拒绝记录和推荐策略
# ---------------------------------------------------------------------------


@dataclass
class AgentMemory:
    """保存跨轮次信息的最小记忆模型。

    preferences 是长期偏好；
    recent_events 是短期轨迹；
    rejected_reasons 用来触发 Reflection；
    consecutive_rejections 用来判断是否连续拒绝了 3 次。
    """

    preferences: dict[str, Any] = field(
        default_factory=lambda: {
            "interests": [],
            "avoid": [],
            "budget": None,
            "must_have": [],
        }
    )
    rejected_reasons: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    accepted_candidates: list[str] = field(default_factory=list)
    recent_events: list[dict[str, str]] = field(default_factory=list)
    strategy: str = "优先满足用户明确表达的兴趣、预算和天气条件。"
    clarifying_question: str = ""
    consecutive_rejections: int = 0
    reflection_count: int = 0
    current_candidate: str = ""

    @classmethod
    def load(cls, path: Path) -> "AgentMemory":
        """从 JSON 文件恢复长期记忆。

        文件不存在或内容损坏时，返回一个全新的空记忆。这样第一次运行不需要
        手动创建文件，后续运行则可以继续使用上一次保存的偏好。
        """

        if not path.exists():
            return cls()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        """将记忆写入本地 JSON 文件，形成一个很简单的持久化记忆。"""

        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_unique(self, key: str, value: str) -> None:
        """向某个偏好列表添加值，并避免重复。"""

        values = self.preferences.setdefault(key, [])
        if value and value not in values:
            values.append(value)

    def update_from_user_message(self, text: str) -> None:
        """从自然语言中提取一些容易识别的偏好。

        这里故意使用规则而不是再调用一次 LLM，目的是展示“数据拆分”：
        用户原话是非结构化文本，函数把它拆成 interests / budget / avoid 三类
        字段，后续 Prompt 只读取结构化结果。

        这是教学版解析器，不可能覆盖所有表达。复杂项目可以将这里替换为
        一个 JSON 输出的 LLM 抽取器或独立信息抽取模型。
        """

        interest_keywords = {
            "历史文化": ("历史", "文化", "古迹", "博物馆", "故宫", "寺庙"),
            "自然风景": ("自然", "风景", "山", "湖", "公园", "徒步"),
            "美食": ("美食", "小吃", "餐厅", "吃"),
            "亲子": ("亲子", "孩子", "儿童", "家庭"),
            "购物": ("购物", "商场", "买东西"),
        }

        for category, keywords in interest_keywords.items():
            if any(keyword in text for keyword in keywords):
                self.add_unique("interests", category)

        # 识别“预算 500 元”或“预算 300-800 元”这样的表达。
        range_match = re.search(
            r"(?:预算|花费|消费)[^\d]{0,10}(\d+)\s*(?:-|~|到)\s*(\d+)\s*元?",
            text,
        )
        single_match = re.search(
            r"(?:预算|花费|消费)[^\d]{0,10}(\d+)\s*元?",
            text,
        )
        if range_match:
            self.preferences["budget"] = {
                "min": int(range_match.group(1)),
                "max": int(range_match.group(2)),
            }
        elif single_match:
            self.preferences["budget"] = {"max": int(single_match.group(1))}

        # 识别“不要太拥挤”“不喜欢购物”等避开项。
        avoid_match = re.findall(
            r"(?:不喜欢|不想要|不要|避免|讨厌|拒绝)([^。！？\n，,]{1,16})",
            text,
        )
        for item in avoid_match:
            cleaned = item.strip(" ：:、 ")
            if cleaned:
                self.add_unique("avoid", cleaned)

        self.recent_events.append({"type": "user_input", "content": text})
        self.recent_events = self.recent_events[-12:]

    def record_rejection(self, candidate: str, reason: str) -> None:
        """记录一次拒绝，并增加连续拒绝计数。"""

        self.consecutive_rejections += 1
        if candidate:
            self.rejected_candidates.append(candidate)
        if reason:
            self.rejected_reasons.append(reason)
        self.recent_events.append(
            {"type": "rejection", "content": reason or "用户拒绝但未说明原因"}
        )
        self.rejected_reasons = self.rejected_reasons[-10:]
        self.rejected_candidates = self.rejected_candidates[-10:]

    def remember_feedback(self, feedback: str) -> None:
        """把用户在推荐之后输入的原话也写入记忆。

        用户反馈不只是“接受/拒绝”标签，其中可能包含新的偏好，例如：
        “我不喜欢太拥挤的地方，但预算可以提高到 800 元”。
        先复用偏好提取器，再把完整原话保存到 recent_events，下一轮 Prompt
        就能同时看到结构化偏好和原始反馈。
        """

        if not feedback:
            return
        self.update_from_user_message(feedback)
        # update_from_user_message 已经记录了一条 user_input 事件；这里把最后
        # 一条事件标记为 user_feedback，便于阅读 agent_memory.json。
        self.recent_events[-1]["type"] = "user_feedback"

    def record_acceptance(self, candidate: str, feedback: str = "") -> None:
        """接受推荐后记录成功案例，并清零“连续拒绝”计数。"""

        if candidate:
            self.accepted_candidates.append(candidate)
            self.accepted_candidates = self.accepted_candidates[-10:]
        self.consecutive_rejections = 0
        self.recent_events.append(
            {
                "type": "acceptance",
                "content": feedback or candidate or "用户接受推荐",
            }
        )

    def context_for_prompt(self) -> str:
        """将内部记忆转换成可读文本，注入到 LLM 的 Thought 阶段。"""

        memory_view = {
            "preferences": self.preferences,
            "rejected_reasons": self.rejected_reasons[-5:],
            "accepted_candidates": self.accepted_candidates[-5:],
            # recent_events 保存用户原话、工具结果、拒绝和反思摘要；这是让
            # LLM “记住 response” 的关键，因为这些内容会被重新放进 Prompt。
            "recent_events": self.recent_events[-8:],
            "strategy": self.strategy,
            "consecutive_rejections": self.consecutive_rejections,
        }
        return json.dumps(memory_view, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 三、Action 解析和工具执行
# ---------------------------------------------------------------------------


@dataclass
class ParsedAction:
    """把模型输出的字符串 Action 拆成结构化字段。"""

    name: str
    kwargs: dict[str, str] = field(default_factory=dict)
    final_answer: str | None = None


def truncate_react_output(text: str) -> str:
    """只保留一组 Thought-Action，避免模型一次输出多组行动。

    `re.DOTALL` 让 `.*?` 可以跨越换行；`?` 让它使用非贪婪模式，尽早在下一组
    Thought / Action / Observation 之前停止。
    """

    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def parse_action(text: str) -> ParsedAction | None:
    """解析两种输出：`Finish[...]` 或 `tool_name(key="value")`。"""

    finish_match = re.search(r"Action:\s*Finish\[(.*?)\]", text, re.DOTALL)
    if finish_match:
        return ParsedAction(name="Finish", final_answer=finish_match.group(1).strip())

    action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", text, re.DOTALL)
    if not action_match:
        return None

    tool_name = action_match.group(1)
    args_text = action_match.group(2)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_text))
    return ParsedAction(name=tool_name, kwargs=kwargs)


def execute_action(action: ParsedAction) -> str:
    """根据解析结果调用真实工具，并把异常转换成 Observation 文本。"""

    tool = AVAILABLE_TOOLS.get(action.name)
    if tool is None:
        return f"错误：未定义的工具 '{action.name}'"

    try:
        return tool(**action.kwargs)
    except TypeError as exc:
        return f"错误：工具参数不正确 - {exc}"
    except Exception as exc:  # 工具网络错误也要回到 Agent 循环中处理
        return f"错误：工具执行失败 - {exc}"


def contains_sold_out_marker(text: str) -> bool:
    """判断工具结果是否包含“门票售罄”的提示。"""

    lowered = text.lower()
    return any(marker in lowered for marker in SOLD_OUT_MARKERS)


def get_alternative_attraction(
    city: str,
    weather: str,
    sold_out_observation: str,
    memory: AgentMemory,
) -> str:
    """票务售罄后的备选循环。

    当前项目没有独立票务 API，所以这里把“避开售罄结果、遵守偏好和拒绝原因”
    拼入景点搜索条件，再调用已有的 Tavily 景点工具。将来接入票务 API 时，
    可以把这个函数替换成：查询可售票景点 -> 过滤 -> 返回 Top 3。
    """

    excluded = "、".join(memory.rejected_candidates[-3:]) or "之前的售罄景点"
    constraints = (
        f"{weather}；请避开已售罄或无法购票的景点（{excluded}）。"
        f"用户偏好：{memory.preferences.get('interests', [])}；"
        f"预算：{memory.preferences.get('budget')}；"
        f"避免：{memory.preferences.get('avoid', [])}。"
    )
    alternative = get_attraction(city, constraints)
    return (
        "原推荐的门票状态为售罄，已自动进入备选循环。\n"
        f"售罄信息：{sold_out_observation}\n"
        f"备选推荐：{alternative}"
    )


# ---------------------------------------------------------------------------
# 四、Reflection：连续拒绝三次后的分析和策略更新
# ---------------------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON，兼容模型偶尔加 Markdown 代码围栏的情况。"""

    cleaned = text.replace("```json", "").replace("```", "").strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        return {}

    try:
        data = json.loads(json_match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def reflect_and_update(client: DeepSeekCompatibleClient, memory: AgentMemory) -> None:
    """执行 Reflection 阶段，并把反思结果写回长期记忆。

    反思输入不是完整聊天记录，而是已经拆分后的拒绝原因、被拒候选和当前策略。
    这样可以减少 Prompt 噪声，也让 Reflection 的职责更清楚：解释失败原因，
    输出下一轮策略，而不是重新生成一个景点。
    """

    reflection_input = {
        "rejected_reasons": memory.rejected_reasons[-5:],
        "rejected_candidates": memory.rejected_candidates[-5:],
        "current_preferences": memory.preferences,
        "current_strategy": memory.strategy,
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(reflection_input, ensure_ascii=False),
                },
            ],
        )
        reflection_text = response.choices[0].message.content or "{}"
        reflection = parse_json_object(reflection_text)
    except Exception as exc:
        # 网络失败时仍然要完成状态迁移，避免 Agent 因 Reflection 失败而卡死。
        reflection = {
            "new_strategy": f"反思服务暂时不可用，先避开已拒绝项目。原因：{exc}",
            "preference_updates": {},
            "clarifying_question": "你更在意预算、距离、环境还是活动类型？",
        }

    updates = reflection.get("preference_updates", {})
    for item in updates.get("interests_add", []):
        memory.add_unique("interests", str(item))
    for item in updates.get("avoid_add", []):
        memory.add_unique("avoid", str(item))
    if updates.get("budget") is not None:
        memory.preferences["budget"] = updates["budget"]

    memory.strategy = str(reflection.get("new_strategy") or memory.strategy)
    memory.clarifying_question = str(reflection.get("clarifying_question") or "")
    memory.reflection_count += 1

    # 反思完成后，连续拒绝计数必须重置。
    # 否则下一轮只拒绝一次就会再次立即触发 Reflection。
    memory.consecutive_rejections = 0
    memory.recent_events.append(
        {"type": "reflection", "content": memory.strategy}
    )

    print("\n[Reflection] 已分析连续拒绝原因并更新推荐策略：")
    print(f"[Reflection] {memory.strategy}")
    if memory.clarifying_question:
        print(f"[Reflection] 后续可主动询问：{memory.clarifying_question}")


# ---------------------------------------------------------------------------
# 五、单次 Thought-Action-Observation 循环
# ---------------------------------------------------------------------------


def build_react_prompt(
    user_input: str,
    memory: AgentMemory,
    short_history: list[str],
) -> str:
    """构建本轮 Prompt。

    这里把输入拆为三块：
    - user_input：本轮用户真正说的话；
    - memory：跨轮次保留的长期信息；
    - short_history：本轮 Thought / Action / Observation 轨迹。
    """

    history = "\n".join(short_history[-8:]) or "（本轮还没有工具调用）"
    return f"""本轮用户输入：
{user_input}

可用长期记忆：
{memory.context_for_prompt()}

本轮短期轨迹：
{history}

请基于以上信息继续执行一个 Thought-Action，或者输出 Finish。
"""


def run_react_cycle(
    client: DeepSeekCompatibleClient,
    memory: AgentMemory,
    user_input: str,
) -> str:
    """运行一次完整的 ReAct 循环，并返回最终答案。"""

    # 入口处先提取偏好，表示图中的“检索记忆模块”。
    memory.update_from_user_message(user_input)
    short_history: list[str] = []

    for step in range(1, MAX_REACT_STEPS + 1):
        print(f"\n--- ReAct 步骤 {step} ---")

        prompt = build_react_prompt(user_input, memory, short_history)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw_output = response.choices[0].message.content or ""
        llm_output = truncate_react_output(raw_output)

        print(f"模型输出：\n{llm_output}")
        short_history.append(llm_output)

        action = parse_action(llm_output)
        if action is None:
            observation = "错误：无法解析 Action，请严格输出 Thought 和 Action。"
            short_history.append(f"Observation: {observation}")
            continue

        if action.name == "Finish":
            answer = action.final_answer or "模型没有提供最终答案。"
            memory.recent_events.append({"type": "finish", "content": answer})
            return answer

        observation = execute_action(action)

        # Action 得到 Observation 后，立刻进行售罄判断。
        # 如果售罄，则不等待下一轮 LLM，而是自动进入备选搜索分支。
        if action.name == "get_attraction" and contains_sold_out_marker(observation):
            city = action.kwargs.get("city", "")
            weather = action.kwargs.get("weather", "")
            observation = get_alternative_attraction(
                city, weather, observation, memory
            )

        if action.name == "get_attraction":
            # 保存本轮候选的文本，用户下一步拒绝时可以把它写入记忆。
            memory.current_candidate = observation

        observation_str = f"Observation: {observation}"
        print(observation_str)
        short_history.append(observation_str)
        memory.recent_events.append({"type": "observation", "content": observation})

    return "本轮思考超过最大步骤，建议补充预算、兴趣或出行范围后再试。"


# ---------------------------------------------------------------------------
# 六、交互主函数：接收反馈、触发 Reflection、保存记忆
# ---------------------------------------------------------------------------


def is_rejection(text: str) -> bool:
    """判断用户反馈是否表示拒绝。

    生产环境可以让 LLM 做更准确的意图分类；教学版先用关键词让逻辑清晰。
    """

    rejection_words = (
        "不喜欢",
        "不要",
        "不想",
        "拒绝",
        "太贵",
        "不合适",
        "不满意",
        "不去",
        "换一个",
        "不考虑",
    )
    return any(word in text for word in rejection_words)


def main() -> None:
    """启动交互式旅行助手。"""

    client = DeepSeekCompatibleClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=BASE_URL,
    )
    memory = AgentMemory.load(MEMORY_FILE)

    print("=== 带记忆和反思的旅行助手 ===")
    print("输入 exit 结束；记忆会保存到 agent_memory.json。")

    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            break
        if not user_input:
            continue

        print("\n[Memory] 检索到的长期记忆：")
        print(memory.context_for_prompt())

        final_answer = run_react_cycle(client, memory, user_input)
        print(f"\n助手：{final_answer}")

        feedback = input(
            "\n请评价本次推荐（接受/拒绝并说明原因），直接回车表示接受："
        ).strip()
        if feedback.lower() in {"exit", "quit", "退出"}:
            memory.save(MEMORY_FILE)
            break

        # 先保存用户反馈本身，再判断它是接受还是拒绝。
        # 这样即使用户说的是新的偏好，也会被提取并带入下一轮。
        memory.remember_feedback(feedback)

        if is_rejection(feedback):
            memory.record_rejection(memory.current_candidate, feedback)
            print(
                f"[Memory] 已记录拒绝原因；连续拒绝次数："
                f"{memory.consecutive_rejections}"
            )

            if memory.consecutive_rejections >= REFLECTION_THRESHOLD:
                # 图中的“连续拒绝次数 >= 3”分支。
                reflect_and_update(client, memory)
        else:
            memory.record_acceptance(memory.current_candidate, feedback)
            print("[Memory] 已记录用户接受本次推荐，连续拒绝计数已清零。")

        # 每轮反馈后落盘，下一次启动仍可以读到偏好和反思策略。
        memory.save(MEMORY_FILE)

        if memory.clarifying_question:
            print(f"[Memory] 主动澄清问题：{memory.clarifying_question}")

        # 用户已经接受当前推荐后，不应该无提示地再次回到“用户：”输入框。
        # 这里对应流程图中的“用户是否继续？”判断：
        # - y / 是：开始下一轮新的旅行请求；
        # - n / 否 / exit：结束本次会话。
        continue_choice = input(
            "\n是否继续？输入 y/是继续，输入 n/否结束："
        ).strip().lower()
        if continue_choice in {"n", "no", "否", "结束", "exit", "quit"}:
            print("会话结束，已保存当前记忆。")
            break


if __name__ == "__main__":
    main()
