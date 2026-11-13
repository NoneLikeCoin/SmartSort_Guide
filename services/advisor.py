# services/advisor.py
"""DeepSeek 智能体：工具调用循环，输入检测列表，输出投放建议。

流程（docs/api.md 第 2 节）：
  1. 校验 DEEPSEEK_API_KEY（只从环境变量读取，绝不硬编码/记录/外发）；
  2. 把 detections + city + scene 交给 deepseek-chat，由模型自主决定
     调用本地工具 map_category / query_rule（实现在 services/rules.py）；
  3. 将工具结果以 role="tool" 回填，循环直至模型给出最终 JSON；
  4. 严格解析并校验最终输出，转换为契约的 items / summary。
"""

import json
import os

from services import rules
from services.rules import RulesError

# ------------------------------------------------------------------ 调用常量

LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
REQUEST_TIMEOUT = 30.0  # 单次请求超时（秒）
MAX_TOOL_ROUNDS = 6  # 工具循环上限，防失控

# ------------------------------------------------------------------ 异常定义


class LLMNotConfiguredError(Exception):
    """未设置 DEEPSEEK_API_KEY，/advise 应降级为 503 LLM_UNAVAILABLE。"""


class LLMResponseError(Exception):
    """调用失败、轮次超限或最终输出不符合契约结构。"""


# ------------------------------------------------------------------ 提示词与工具

SYSTEM_PROMPT = (
    "你是垃圾分类投放指导智能体。用户会给出若干垃圾检测项（class 为细粒度英文类别）、"
    "投放城市 city 与场景 scene。规则：\n"
    "1. 对每一个不同的 class，必须先调用 map_category 得到中文四分类；\n"
    "2. 再针对每个四分类调用 query_rule（传入该四分类、用户给的 city 和 scene），"
    "获取投放要求、桶颜色与原因；\n"
    "3. 完成工具调用后，只输出一个 JSON 对象，不要输出任何其他文字或 Markdown 代码块：\n"
    '{"items": [{"class": "<英文类别>", "category": "<中文四分类>", '
    '"bin": "<桶颜色与名称>", "reason": "<不超过80字的投放说明，结合工具结果与场景>"}], '
    '"summary": "<不超过120字的整体说明：共几件、按四分类各几件、一句投放提醒>"}\n'
    "items 与用户 detections 逐条对应（同一 class 出现多次则输出多条）；"
    "只依据工具返回的规则作答，不要编造政策。"
)

# 工具 schema：参数枚举由规则库动态生成，保证与数据表一致
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "map_category",
            "description": "将细粒度垃圾英文类别映射为中文四分类（可回收物/有害垃圾/厨余垃圾/其他垃圾）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "enum": sorted(rules.allowed_classes()),
                        "description": "检测输出的英文类别标签",
                    }
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_rule",
            "description": "查询某个中文四分类在指定城市与场景下的投放要求、桶颜色、原因说明与本地注意事项。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"],
                        "description": "中文四分类",
                    },
                    "city": {
                        "type": "string",
                        "enum": ["changsha", "beijing", "shanghai"],
                        "description": "投放城市",
                    },
                    "scene": {
                        "type": "string",
                        "enum": ["campus", "residential", "office"],
                        "description": "投放场景",
                    },
                },
                "required": ["category", "city", "scene"],
            },
        },
    },
]


def _create_client(api_key):
    """构造 DeepSeek 兼容客户端。单独成函数便于测试替换。"""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL, timeout=REQUEST_TIMEOUT)


# ------------------------------------------------------------------ 工具执行


def _run_tool(name, arguments, city, scene):
    """执行本地工具并返回可 JSON 序列化的结果；表外查询以 error 返回给模型纠错。"""
    try:
        if name == "map_category":
            return rules.map_category(arguments.get("class_name", ""))
        if name == "query_rule":
            return rules.query_rule(
                arguments.get("category", ""),
                arguments.get("city") or city,
                arguments.get("scene") or scene,
            )
    except RulesError as exc:
        return {"error": str(exc)}
    return {"error": "未知工具：{}".format(name)}


# ------------------------------------------------------------------ 输出校验


def _parse_final(content, detections):
    """把模型最终文本严格校验为 (items, summary)。不符合契约则抛 LLMResponseError。"""
    if not content or not content.strip():
        raise LLMResponseError("智能体最终输出为空。")

    text = content.strip()
    # 容错：剥离可能包裹 JSON 的 Markdown 代码围栏（首行 ``` 或 ```json，尾行 ```）
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LLMResponseError("最终输出不是合法 JSON：{}".format(exc)) from exc

    if not isinstance(data, dict):
        raise LLMResponseError("最终输出不是 JSON 对象。")

    allowed = rules.allowed_classes()
    raw_items = data.get("items")
    summary = data.get("summary")
    if not isinstance(raw_items, list) or not raw_items:
        raise LLMResponseError("items 缺失或为空。")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMResponseError("summary 缺失或为空。")

    items = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise LLMResponseError("items 的每一项都必须是对象。")
        if item.get("class") not in allowed:
            raise LLMResponseError("items 含契约外类别：{}".format(item.get("class")))
        for key in ("category", "bin", "reason"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise LLMResponseError("items 缺少合法字段：{}".format(key))
        items.append(
            {
                "class": item["class"],
                "category": item["category"],
                "bin": item["bin"],
                "reason": item["reason"],
            }
        )

    # 每个被检测类别至少应有一条建议，缺失视为输出不完整
    expected = {det["class"] for det in detections}
    covered = {item["class"] for item in items}
    if not expected.issubset(covered):
        raise LLMResponseError(
            "items 未覆盖全部检测类别，缺少：{}".format(sorted(expected - covered))
        )

    return items, summary.strip()


# ------------------------------------------------------------------ 对外接口


def get_advise(detections, city, scene):
    """调用 DeepSeek 智能体，返回 (items, summary)，字段严格符合契约第 2 节。

    可能抛出：LLMNotConfiguredError（密钥缺失）/ LLMResponseError（调用或输出异常）。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise LLMNotConfiguredError("未设置 DEEPSEEK_API_KEY 环境变量。")

    client = _create_client(api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"detections": detections, "city": city, "scene": scene},
                ensure_ascii=False,
            ),
        },
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0.2,
            )
        except Exception as exc:
            # openai 异常种类多（鉴权/网络/限流），统一收敛；
            # 不打印密钥，异常文本本身不含 key。
            raise LLMResponseError("调用 DeepSeek 失败：{}".format(exc)) from exc

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except ValueError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                result = _run_tool(tc.function.name, arguments, city, scene)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        return _parse_final(msg.content, detections)

    raise LLMResponseError("工具调用超过 {} 轮仍未产出最终结果。".format(MAX_TOOL_ROUNDS))
