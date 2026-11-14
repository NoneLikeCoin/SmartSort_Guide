# tests/test_advisor.py
"""DeepSeek 智能体循环的离线测试：伪造 API 客户端，不发起任何网络请求。

覆盖：密钥缺失守卫、多轮工具调用（assistant/tool 消息回填）、
规则库真实执行结果、最终 JSON 严格校验与容错、轮次上限与调用异常收敛。
"""

import json
from types import SimpleNamespace

import pytest

from services import advisor
from services.advisor import get_advise, LLMNotConfiguredError, LLMResponseError

DETECTIONS = [
    {"class": "plastic", "confidence": 0.9},
    {"class": "trash", "confidence": 0.6},
]

FINAL_ITEMS = {
    "items": [
        {
            "class": "plastic",
            "category": "可回收物",
            "bin": "蓝色可回收物桶",
            "reason": "塑料瓶属于可回收物，投放前请清空内容物并压扁。",
        },
        {
            "class": "trash",
            "category": "其他垃圾",
            "bin": "灰色其他垃圾桶",
            "reason": "其他垃圾难以回收利用，请投入灰色其他垃圾桶。",
        },
    ],
    "summary": "本次共检测到 2 件垃圾：可回收物 1 件、其他垃圾 1 件，请分类投放。",
}


def tool_response(calls):
    """构造一条含 tool_calls 的模型响应（结构对齐 openai SDK 返回对象）。"""
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call["id"],
                function=SimpleNamespace(
                    name=call["name"], arguments=json.dumps(call["args"])
                ),
            )
            for call in calls
        ],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def final_response(payload_text):
    message = SimpleNamespace(content=payload_text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []  # 记录每次 create 的 kwargs，便于断言消息回填

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dummy-offline-test")


@pytest.fixture
def fake_client(monkeypatch):
    """把 responses 塞进假客户端并替换 advisor._create_client。"""

    def _install(responses):
        fake = FakeClient(responses)
        monkeypatch.setattr(advisor, "_create_client", lambda key: fake)
        return fake

    return _install


# ------------------------------------------------------------------ 守卫


def test_missing_api_key_raises(with_key, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(LLMNotConfiguredError):
        get_advise(DETECTIONS, "changsha", "campus")


# ------------------------------------------------------------------ 工具循环主链路


def test_tool_loop_executes_rules_and_returns_contract_shape(with_key, fake_client):
    fake = fake_client(
        [
            tool_response(
                [
                    {"id": "c1", "name": "map_category", "args": {"class_name": "plastic"}},
                    {"id": "c2", "name": "map_category", "args": {"class_name": "trash"}},
                ]
            ),
            tool_response(
                [
                    {"id": "c3", "name": "query_rule",
                     "args": {"category": "可回收物", "city": "changsha", "scene": "campus"}},
                    {"id": "c4", "name": "query_rule",
                     "args": {"category": "其他垃圾", "city": "changsha", "scene": "campus"}},
                ]
            ),
            final_response(json.dumps(FINAL_ITEMS, ensure_ascii=False)),
        ]
    )

    items, summary = get_advise(DETECTIONS, "changsha", "campus")

    # 输出形状严格符合契约第 2 节
    assert items == FINAL_ITEMS["items"]
    assert summary == FINAL_ITEMS["summary"]

    # 会话记录：最后一轮请求携带完整历史
    messages = fake.chat.completions.calls[-1]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "tool", "tool"]

    # 工具结果按 tool_call_id 一一对应回填
    tool_messages = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert set(tool_messages) == {"c1", "c2", "c3", "c4"}
    assert json.loads(tool_messages["c1"]) == "可回收物"
    assert json.loads(tool_messages["c2"]) == "其他垃圾"
    rule_c3 = json.loads(tool_messages["c3"])
    assert rule_c3["bin"] == "蓝色可回收物桶"
    assert rule_c3["requirement"] and rule_c3["reason"]
    assert "长沙市" in rule_c3["note"]  # 城市说明来自规则库
    assert "校园" in rule_c3["note"] or "回收驿站" in rule_c3["note"]  # 场景说明

    # assistant 历史消息携带 tool_calls 结构（openai 协议要求）
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "map_category"

    # 调用参数：模型名与工具 schema 正确传入
    kwargs = fake.chat.completions.calls[0]
    assert kwargs["model"] == advisor.LLM_MODEL
    tool_names = {t["function"]["name"] for t in kwargs["tools"]}
    assert tool_names == {"map_category", "query_rule"}


def test_unknown_tool_class_returns_error_to_model(with_key, fake_client):
    """表外类别：工具以 error 内容回填，让模型自我纠错而非崩溃。"""
    fake = fake_client(
        [
            tool_response([{"id": "c1", "name": "map_category", "args": {"class_name": "banana"}}]),
            final_response(json.dumps(FINAL_ITEMS, ensure_ascii=False)),
        ]
    )
    items, _ = get_advise(DETECTIONS, "changsha", "campus")
    messages = fake.chat.completions.calls[-1]["messages"]
    tool_content = [m for m in messages if m["role"] == "tool"][0]["content"]
    assert "未知类别" in json.loads(tool_content)["error"]
    assert len(items) == 2


# ------------------------------------------------------------------ 最终输出校验


def test_markdown_fenced_output_is_parsed(with_key, fake_client):
    fenced = "```json\n" + json.dumps(FINAL_ITEMS, ensure_ascii=False) + "\n```"
    fake_client([final_response(fenced)])
    items, summary = get_advise(DETECTIONS, "changsha", "campus")
    assert len(items) == 2 and summary


def test_first_call_forces_tool_choice(with_key, fake_client):
    """结构性保证：第一次调用必须 tool_choice=required（禁止单轮直接作答），
    之后交还模型自主（auto）。"""
    fake = fake_client(
        [
            tool_response([{"id": "c1", "name": "map_category", "args": {"class_name": "plastic"}}]),
            final_response(json.dumps(FINAL_ITEMS, ensure_ascii=False)),
        ]
    )
    get_advise(DETECTIONS, "changsha", "campus")
    choices = [c["tool_choice"] for c in fake.chat.completions.calls]
    assert choices == [advisor.FIRST_ROUND_TOOL_CHOICE, advisor.FOLLOWUP_TOOL_CHOICE]
    assert choices[0] == "required"


def test_correction_round_recovers_from_bad_output(with_key, fake_client):
    """自我修正：最终输出非法 -> 把校验错误发回模型 -> 第二次重写成功。"""
    fake = fake_client(
        [
            final_response("建议您把塑料瓶投进可回收物桶哦。"),  # 非法：纯文本
            final_response(json.dumps(FINAL_ITEMS, ensure_ascii=False)),  # 修正后合法
        ]
    )
    items, summary = get_advise(DETECTIONS, "changsha", "campus")
    assert len(items) == 2 and summary

    assert len(fake.chat.completions.calls) == 2
    last_messages = fake.chat.completions.calls[-1]["messages"]
    # 修正轮历史：非法输出作为 assistant 保留，错误说明作为 user 反馈
    correction = [
        m for m in last_messages
        if m["role"] == "user" and "不符合要求" in m["content"]
    ]
    assert correction and "JSON" in correction[0]["content"]


def test_bad_output_twice_raises(with_key, fake_client):
    """修正轮用尽仍非法：抛 LLMResponseError（app 层转 500）。"""
    fake = fake_client(
        [final_response("今天天气不错"), final_response("还是不想输出 JSON")]
    )
    with pytest.raises(LLMResponseError) as excinfo:
        get_advise(DETECTIONS, "changsha", "campus")
    assert "合法 JSON" in str(excinfo.value)
    assert len(fake.chat.completions.calls) == 1 + advisor.MAX_CORRECTIONS


def test_missing_class_coverage_raises(with_key, fake_client):
    partial = {"items": [FINAL_ITEMS["items"][0]], "summary": FINAL_ITEMS["summary"]}
    fake_client([final_response(json.dumps(partial, ensure_ascii=False))] * 2)
    with pytest.raises(LLMResponseError):
        get_advise(DETECTIONS, "changsha", "campus")


def test_item_with_illegal_class_raises(with_key, fake_client):
    bad = json.loads(json.dumps(FINAL_ITEMS))
    bad["items"][0]["class"] = "banana"
    fake_client([final_response(json.dumps(bad, ensure_ascii=False))] * 2)
    with pytest.raises(LLMResponseError):
        get_advise(DETECTIONS, "changsha", "campus")


# ------------------------------------------------------------------ 异常与上限


def test_round_limit_exceeded_raises(with_key, fake_client):
    looping = [
        tool_response([{"id": "c1", "name": "map_category", "args": {"class_name": "plastic"}}])
        for _ in range(advisor.MAX_TOOL_ROUNDS + 2)
    ]
    fake_client(looping)
    with pytest.raises(LLMResponseError):
        get_advise(DETECTIONS, "changsha", "campus")


def test_api_failure_wrapped_as_llm_response_error(with_key, monkeypatch):
    class BoomCompletions:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=BoomCompletions()))
    monkeypatch.setattr(advisor, "_create_client", lambda key: fake)
    with pytest.raises(LLMResponseError) as excinfo:
        get_advise(DETECTIONS, "changsha", "campus")
    assert "DeepSeek" in str(excinfo.value)
    # 密钥绝不随异常信息外泄
    assert "sk-dummy-offline-test" not in str(excinfo.value)
