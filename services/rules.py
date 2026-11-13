# services/rules.py
"""类别映射表与投放规则查询 —— 供智能体调用的本地工具函数。

规则库默认读取项目内 data/rules.json；生产环境可通过环境变量
RULES_PATH 指向挂载路径（该文件不进 Docker 镜像）。
"""

import json
import os

# 项目根目录：services/ 的上一级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_RULES_PATH = os.path.join(_PROJECT_ROOT, "data", "rules.json")

# 规则库必需字段
_REQUIRED_KEYS = ("category_map", "rules", "city_notes", "scene_notes")

_cache = None


class RulesError(Exception):
    """规则库缺失、损坏，或查询了表外的类别/分类。"""


def _load():
    """读取并缓存规则库；配置变化需重启进程生效（与模型加载策略一致）。"""
    global _cache
    if _cache is None:
        path = os.environ.get("RULES_PATH", "").strip() or _DEFAULT_RULES_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            raise RulesError("规则库文件不可读：{}".format(exc)) from exc
        except ValueError as exc:
            raise RulesError("规则库不是合法 JSON：{}".format(exc)) from exc
        missing = [k for k in _REQUIRED_KEYS if not isinstance(data.get(k), dict)]
        if missing:
            raise RulesError("规则库缺少字段：{}".format(missing))
        _cache = data
    return _cache


def allowed_classes():
    """契约第 5 节允许的全部类别标签集合。"""
    return set(_load()["category_map"].keys())


def map_category(class_name):
    """工具一：细粒度类别 -> 中文四分类（可回收物/有害垃圾/厨余垃圾/其他垃圾）。"""
    category = _load()["category_map"].get(str(class_name).strip().lower())
    if category is None:
        raise RulesError("未知类别：{}".format(class_name))
    return category


def query_rule(category, city="", scene=""):
    """工具二：四分类 + 城市 + 场景 -> 投放要求、桶颜色与原因说明。

    城市/场景说明为附加参考（note），未收录的取值留空即可，
    智能体仍可基于基础规则作答。
    """
    data = _load()
    rule = data["rules"].get(str(category).strip())
    if rule is None:
        raise RulesError("未知四分类：{}".format(category))
    notes = [data["city_notes"].get(city, ""), data["scene_notes"].get(scene, "")]
    return {
        "category": category,
        "bin": rule["bin"],
        "requirement": rule["requirement"],
        "reason": rule["reason"],
        "note": " ".join(p for p in notes if p),
    }
