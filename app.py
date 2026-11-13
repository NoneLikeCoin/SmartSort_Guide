# app.py
# -*- coding: utf-8 -*-
"""智能垃圾分类识别与投放指导系统 —— Flask 入口。

职责边界：
  1. 只负责路由注册、请求参数校验、统一错误响应；
  2. MOCK=1 时使用 _mock_* 函数返回固定数据（不加载模型、不调用外部 API）；
  3. MOCK!=1 时检测与建议接口直接返回 503，提示能力尚未就绪。
"""

import os
import time
import uuid

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ------------------------------------------------------------------ 基础配置

# 图片体积上限：10MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# 整体请求体上限比图片上限略大，用于容纳 multipart 边界等额外开销，
# 真正的图片大小校验在视图函数里精确执行。
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024

# ------------------------------------------------------------------ 常量定义

# 允许上传的图片扩展名
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# 允许的投放城市
ALLOWED_CITIES = {"changsha", "beijing", "shanghai"}

# 允许的投放场景
ALLOWED_SCENES = {"campus", "residential", "office"}

# 类别标签（本系统只允许这 6 类）
CLASS_LABELS = {
    "cardboard": "纸板",
    "glass": "玻璃",
    "metal": "金属",
    "paper": "纸",
    "plastic": "塑料",
    "trash": "其他垃圾",
}

# 各类别对应的投放建议（mock 阶段的固定映射，字段与 /advise 响应一致）
CLASS_ADVICE = {
    "cardboard": {
        "category": "可回收物",
        "bin": "蓝色可回收物桶",
        "reason": "纸板属于可回收物，请拆开压平后投入蓝色可回收物桶。",
    },
    "glass": {
        "category": "可回收物",
        "bin": "蓝色可回收物桶",
        "reason": "玻璃制品可回收再利用，请小心包裹后投入蓝色可回收物桶。",
    },
    "metal": {
        "category": "可回收物",
        "bin": "蓝色可回收物桶",
        "reason": "金属可回收熔炼再利用，请清空内容物后投入蓝色可回收物桶。",
    },
    "paper": {
        "category": "可回收物",
        "bin": "蓝色可回收物桶",
        "reason": "纸张属于可回收物，保持干燥平整后投入蓝色可回收物桶。",
    },
    "plastic": {
        "category": "可回收物",
        "bin": "蓝色可回收物桶",
        "reason": "塑料瓶罐请倒空液体、简单冲洗并压扁后投入蓝色可回收物桶。",
    },
    "trash": {
        "category": "其他垃圾",
        "bin": "灰色其他垃圾桶",
        "reason": "该类垃圾难以回收利用，请投入灰色其他垃圾桶。",
    },
}

# ------------------------------------------------------------------ 通用工具


def _error(code, message, status):
    """构造统一的错误响应体：{"error": {"code": ..., "message": ...}}。"""
    return jsonify({"error": {"code": code, "message": message}}), status


def _is_mock():
    """当前是否运行在 mock 模式（环境变量 MOCK=1）。"""
    return os.environ.get("MOCK", "").strip() == "1"


def _new_request_id():
    """生成一次请求的追踪 ID。"""
    return uuid.uuid4().hex


def _elapsed_ms(start_ts):
    """计算从 start_ts 到现在的毫秒数。"""
    return int(round((time.time() - start_ts) * 1000))


def _file_extension(filename):
    """取文件名的小写扩展名；无扩展名时返回空串。"""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def _looks_like_image(raw, ext):
    """通过文件头魔数粗判图片内容是否与扩展名匹配。"""
    if not raw:
        return False
    if ext == "png":
        return raw.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in ("jpg", "jpeg"):
        return raw.startswith(b"\xff\xd8\xff")
    return False


# ------------------------------------------------------------------ mock 数据


def _mock_detect(filename):
    """mock 检测：返回固定的 2 条检测结果，坐标均落在 640x480 之内。

    参数 filename 仅用于占位，真实模型接入后可据此做日志记录。
    """
    _ = filename  # 当前 mock 阶段不使用
    return {
        "detections": [
            {"class": "plastic", "confidence": 0.87, "box": [120, 80, 340, 410]},
            {"class": "paper", "confidence": 0.74, "box": [365, 150, 520, 300]},
        ],
        "image_size": {"width": 640, "height": 480},
    }


def _mock_advise(detections, city, scene):
    """mock 建议：按类别固定映射生成投放建议，不调用任何外部大模型。

    参数 city / scene 仅用于占位，真实大模型接入后可据此做地区化、场景化措辞。
    """
    _ = (city, scene)  # 当前 mock 阶段不使用

    items = []
    for det in detections:
        advice = CLASS_ADVICE[det["class"]]
        items.append(
            {
                "class": det["class"],
                "category": advice["category"],
                "bin": advice["bin"],
                "reason": advice["reason"],
            }
        )

    # 按四分类统计件数，拼出一句汇总说明
    counts = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    parts = ["{} {} 件".format(category, count) for category, count in counts.items()]

    if items:
        summary = "本次共识别 {} 件垃圾，其中{}。请按下方建议分类投放。".format(
            len(items), "、".join(parts)
        )
    else:
        summary = "本次未识别到垃圾。"

    return {"items": items, "summary": summary}


# ------------------------------------------------------------------ 路由


@app.route("/", methods=["GET"])
def index():
    """单页界面。"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """健康检查：mock 阶段不加载模型、不配置大模型。"""
    return jsonify(
        {
            "status": "ok",
            "mock": _is_mock(),
            "model_loaded": False,
            "llm_configured": False,
        }
    )


@app.route("/detect", methods=["POST"])
def detect():
    """图片识别接口：接收 multipart/form-data 的 image 字段。"""
    start_ts = time.time()
    request_id = _new_request_id()

    # ---- 参数校验：文件字段是否存在 ----
    upload = request.files.get("image")
    if upload is None:
        return _error("MISSING_IMAGE", "请求中缺少 image 文件字段。", 400)
    if not upload.filename:
        return _error("MISSING_IMAGE", "未选择需要识别的图片文件。", 400)

    # ---- 参数校验：扩展名 / 媒体类型 ----
    ext = _file_extension(upload.filename)
    if ext not in ALLOWED_EXTENSIONS:
        return _error("UNSUPPORTED_MEDIA_TYPE", "仅支持 jpg、jpeg、png 格式的图片。", 415)

    # ---- 参数校验：体积与内容有效性 ----
    raw = upload.read()
    if len(raw) > MAX_IMAGE_BYTES:
        return _error("IMAGE_TOO_LARGE", "图片大小超过 10MB 限制。", 413)
    if not _looks_like_image(raw, ext):
        return _error("INVALID_IMAGE", "图片无法解析，请上传有效的 jpg 或 png 图片。", 400)

    # ---- 非 mock 模式：模型尚未接入 ----
    if not _is_mock():
        return _error("MODEL_NOT_READY", "识别模型尚未就绪，请稍后重试。", 503)

    # ---- mock 模式：返回固定检测结果 ----
    try:
        result = _mock_detect(upload.filename)
    except Exception:  # 兜底，避免内部异常信息外泄
        return _error("DETECT_FAILED", "图片识别失败，请重试。", 500)

    return jsonify(
        {
            "request_id": request_id,
            "detections": result["detections"],
            "image_size": result["image_size"],
            "elapsed_ms": _elapsed_ms(start_ts),
        }
    )


@app.route("/advise", methods=["POST"])
def advise():
    """投放指导接口：接收检测结果 + 城市 + 场景，返回四分类投放建议。"""
    start_ts = time.time()
    request_id = _new_request_id()

    # ---- 参数校验：请求体必须是 JSON 对象 ----
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("INVALID_PARAM", "请求体必须是合法的 JSON 对象。", 400)

    # ---- 参数校验：detections 非空 ----
    detections = payload.get("detections")
    if not isinstance(detections, list) or not detections:
        return _error("EMPTY_DETECTIONS", "detections 不能为空，请先完成图片识别。", 400)

    # ---- 参数校验：city / scene 取值 ----
    city = payload.get("city")
    if not isinstance(city, str) or city not in ALLOWED_CITIES:
        return _error("INVALID_PARAM", "city 只能是 changsha、beijing 或 shanghai。", 400)

    scene = payload.get("scene")
    if not isinstance(scene, str) or scene not in ALLOWED_SCENES:
        return _error("INVALID_PARAM", "scene 只能是 campus、residential 或 office。", 400)

    # ---- 参数校验：逐条检测结果的类别与置信度 ----
    clean_detections = []
    for det in detections:
        if not isinstance(det, dict):
            return _error("INVALID_PARAM", "detections 的每一项都必须是对象。", 400)

        class_name = det.get("class")
        if class_name not in CLASS_LABELS:
            return _error(
                "UNKNOWN_CLASS",
                "存在不支持的类别标签：{}。".format(class_name),
                400,
            )

        confidence = det.get("confidence", 0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return _error("INVALID_PARAM", "confidence 必须是数字。", 400)
        if confidence < 0 or confidence > 1:
            return _error("INVALID_PARAM", "confidence 必须位于 0 到 1 之间。", 400)

        clean_detections.append({"class": class_name, "confidence": float(confidence)})

    # ---- 非 mock 模式：大模型尚未配置 ----
    if not _is_mock():
        return _error("LLM_UNAVAILABLE", "投放建议服务尚未配置，请稍后重试。", 503)

    # ---- mock 模式：返回固定映射建议 ----
    try:
        result = _mock_advise(clean_detections, city, scene)
    except Exception:  # 兜底，避免内部异常信息外泄
        return _error("ADVISE_FAILED", "生成投放建议失败，请重试。", 500)

    return jsonify(
        {
            "request_id": request_id,
            "items": result["items"],
            "summary": result["summary"],
            "elapsed_ms": _elapsed_ms(start_ts),
        }
    )


# ------------------------------------------------------------------ 错误处理


@app.errorhandler(413)
def _handle_413(_exc):
    """请求体超过 MAX_CONTENT_LENGTH 时统一包装。"""
    return _error("IMAGE_TOO_LARGE", "图片大小超过 10MB 限制。", 413)


@app.errorhandler(404)
def _handle_404(_exc):
    """接口不存在。"""
    return _error("NOT_FOUND", "请求的接口不存在。", 404)


@app.errorhandler(405)
def _handle_405(_exc):
    """请求方法不被允许。"""
    return _error("METHOD_NOT_ALLOWED", "请求方法不被允许。", 405)


@app.errorhandler(500)
def _handle_500(_exc):
    """服务器内部错误兜底。"""
    return _error("INTERNAL_ERROR", "服务器内部错误，请稍后重试。", 500)


if __name__ == "__main__":
    # 本地开发入口；生产环境请使用 WSGI 服务器托管
    app.run(host="0.0.0.0", port=5000)