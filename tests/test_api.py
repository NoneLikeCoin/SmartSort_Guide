# tests/test_api.py
"""接口契约测试：mock 链路、503 降级、参数校验、/health 真实探测。

契约来源 docs/api.md：字段名 / 类型 / 错误码。
"""

import io
import os

import app as app_module

CONTRACT_DETECT_KEYS = {"request_id", "detections", "image_size", "elapsed_ms"}
CONTRACT_ADVISE_KEYS = {"request_id", "items", "summary", "elapsed_ms"}
CONTRACT_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}


def post_detect(client, image_bytes, filename="photo.png"):
    return client.post(
        "/detect",
        data={"image": (io.BytesIO(image_bytes), filename)},
        content_type="multipart/form-data",
    )


# ------------------------------------------------------------------ mock 正常链路


class TestMockChain:
    def test_detect_mock(self, client, monkeypatch, valid_png_bytes):
        """MOCK=1：/detect 返回契约定义的成功结构。"""
        monkeypatch.setenv("MOCK", "1")
        resp = post_detect(client, valid_png_bytes)
        assert resp.status_code == 200
        data = resp.get_json()

        assert set(data) == CONTRACT_DETECT_KEYS
        assert isinstance(data["request_id"], str) and data["request_id"]
        assert isinstance(data["elapsed_ms"], int) and data["elapsed_ms"] >= 0
        assert set(data["image_size"]) == {"width", "height"}
        assert data["image_size"]["width"] > 0 and data["image_size"]["height"] > 0

        assert isinstance(data["detections"], list) and data["detections"]
        for det in data["detections"]:
            assert set(det) == {"class", "confidence", "box"}
            assert det["class"] in CONTRACT_CLASSES
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0
            assert isinstance(det["box"], list) and len(det["box"]) == 4
            assert det["box"][0] <= det["box"][2] and det["box"][1] <= det["box"][3]

    def test_mock_full_chain_detect_then_advise(
        self, client, monkeypatch, valid_png_bytes
    ):
        """MOCK=1：/detect → /advise 端到端，字段严格符合契约第 2 节。"""
        monkeypatch.setenv("MOCK", "1")
        detections = post_detect(client, valid_png_bytes).get_json()["detections"]

        resp = client.post(
            "/advise",
            json={"detections": detections, "city": "changsha", "scene": "campus"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert set(data) == CONTRACT_ADVISE_KEYS
        assert isinstance(data["summary"], str) and data["summary"]
        assert len(data["items"]) == len(detections)
        for item in data["items"]:
            assert set(item) == {"class", "category", "bin", "reason"}
            assert item["class"] in CONTRACT_CLASSES
            assert all(isinstance(item[k], str) and item[k] for k in ("category", "bin", "reason"))


# ------------------------------------------------------------------ 真实模式 503


class TestDetectRealMode:
    def test_detect_no_model_503(self, client, valid_png_bytes):
        """MOCK=0 且未设置 MODEL_PATH：503 MODEL_NOT_READY，错误体为统一格式。"""
        resp = post_detect(client, valid_png_bytes)
        assert resp.status_code == 503
        body = resp.get_json()
        assert set(body) == {"error"}
        assert body["error"]["code"] == "MODEL_NOT_READY"
        assert isinstance(body["error"]["message"], str)

    def test_detect_model_path_not_found_503(self, client, monkeypatch, valid_png_bytes):
        """MODEL_PATH 指向不存在的文件：同样降级 503，不崩溃。"""
        monkeypatch.setenv("MODEL_PATH", "Z:/no/such/best.onnx")
        resp = post_detect(client, valid_png_bytes)
        assert resp.status_code == 503
        assert resp.get_json()["error"]["code"] == "MODEL_NOT_READY"

    def test_detect_model_path_invalid_onnx_503(
        self, client, monkeypatch, tmp_path, valid_png_bytes
    ):
        """MODEL_PATH 存在但不是合法 ONNX：加载异常被捕获，返回 503 而非 500。"""
        fake = tmp_path / "broken.onnx"
        fake.write_bytes(b"definitely not an onnx model")
        monkeypatch.setenv("MODEL_PATH", str(fake))
        resp = post_detect(client, valid_png_bytes)
        assert resp.status_code == 503
        assert resp.get_json()["error"]["code"] == "MODEL_NOT_READY"

    def test_advise_without_api_key_503(self, client):
        """MOCK=0 且无 DEEPSEEK_API_KEY：/advise 返回 503 LLM_UNAVAILABLE（不发起网络调用）。"""
        resp = client.post(
            "/advise",
            json={
                "detections": [{"class": "plastic", "confidence": 0.9}],
                "city": "changsha",
                "scene": "campus",
            },
        )
        assert resp.status_code == 503
        assert resp.get_json()["error"]["code"] == "LLM_UNAVAILABLE"


# ------------------------------------------------------------------ 参数校验与错误码


class TestDetectValidation:
    def test_invalid_image_bytes_400(self, client, monkeypatch):
        """扩展名合规但内容无法解码：400 INVALID_IMAGE。"""
        monkeypatch.setenv("MOCK", "1")
        resp = post_detect(client, b"this is not an image", filename="fake.png")
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "INVALID_IMAGE"

    def test_unsupported_extension_415(self, client, monkeypatch, valid_png_bytes):
        """不支持的扩展名：415 UNSUPPORTED_MEDIA_TYPE。"""
        monkeypatch.setenv("MOCK", "1")
        resp = post_detect(client, valid_png_bytes, filename="notes.txt")
        assert resp.status_code == 415
        assert resp.get_json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_missing_image_field_400(self, client, monkeypatch):
        """缺少 image 字段：400 MISSING_IMAGE。"""
        monkeypatch.setenv("MOCK", "1")
        resp = client.post("/detect", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "MISSING_IMAGE"


class TestAdviseValidation:
    def _payload(self, **overrides):
        payload = {
            "detections": [{"class": "plastic", "confidence": 0.87}],
            "city": "changsha",
            "scene": "campus",
        }
        payload.update(overrides)
        return payload

    def test_empty_detections_400(self, client, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        resp = client.post("/advise", json=self._payload(detections=[]))
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "EMPTY_DETECTIONS"

    def test_unknown_class_400(self, client, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        resp = client.post(
            "/advise",
            json=self._payload(detections=[{"class": "banana", "confidence": 0.5}]),
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "UNKNOWN_CLASS"

    def test_confidence_out_of_range_400(self, client, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        resp = client.post(
            "/advise",
            json=self._payload(detections=[{"class": "plastic", "confidence": 1.5}]),
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "INVALID_PARAM"

    def test_invalid_city_400(self, client, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        resp = client.post("/advise", json=self._payload(city="guangzhou"))
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "INVALID_PARAM"

    def test_non_json_body_400(self, client, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        resp = client.post("/advise", data="hello", content_type="text/plain")
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "INVALID_PARAM"


# ------------------------------------------------------------------ /health 探测


class TestHealth:
    def test_health_mock_mode(self, client, monkeypatch):
        """MOCK=1：一律报告未就绪（不加载模型、不使用大模型）。"""
        monkeypatch.setenv("MOCK", "1")
        data = client.get("/health").get_json()
        assert data == {
            "status": "ok",
            "mock": True,
            "model_loaded": False,
            "llm_configured": False,
        }

    def test_health_real_without_anything(self, client):
        data = client.get("/health").get_json()
        assert data["mock"] is False
        assert data["model_loaded"] is False
        assert data["llm_configured"] is False

    def test_health_model_file_present(self, client, monkeypatch, tmp_path):
        """真实模式：MODEL_PATH 文件存在时 model_loaded 反映为 true。"""
        model = tmp_path / "best.onnx"
        model.write_bytes(b"x")
        monkeypatch.setenv("MODEL_PATH", str(model))
        data = client.get("/health").get_json()
        assert data["model_loaded"] is True

    def test_health_llm_configured_never_leaks_key(self, client, monkeypatch):
        """真实模式：配置密钥后 llm_configured 为 true，且响应不含密钥本身。"""
        secret = "sk-secret-must-not-appear"
        monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
        resp = client.get("/health")
        data = resp.get_json()
        assert data["llm_configured"] is True
        assert secret not in resp.get_data(as_text=True)


# ------------------------------------------------------------------ .env 加载


class TestDotenv:
    def _write_env(self, tmp_path, text):
        env_file = tmp_path / ".env"
        env_file.write_text(text, encoding="utf-8")
        return str(env_file)

    def test_parses_and_fills_missing(self, tmp_path):
        """.env 中的 KEY=VALUE 写入环境；注释、空行、非法行跳过；引号剥除。"""
        path = self._write_env(
            tmp_path,
            "# 注释行\n"
            "MOCK=1\n"
            "MODEL_PATH='D:/models/best.onnx'\n"
            "\n"
            'DEEPSEEK_API_KEY="sk-from-dotenv"\n'
            "这一行没有等号\n",
        )
        app_module._load_dotenv(path)
        assert os.environ["MOCK"] == "1"
        assert os.environ["MODEL_PATH"] == "D:/models/best.onnx"
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-dotenv"

    def test_system_env_wins_over_dotenv(self, monkeypatch, tmp_path):
        """系统环境变量优先：.env 不得覆盖已存在的变量。"""
        monkeypatch.setenv("MOCK", "0")
        path = self._write_env(tmp_path, "MOCK=1\n")
        app_module._load_dotenv(path)
        assert os.environ["MOCK"] == "0"

    def test_missing_file_is_noop(self, tmp_path):
        """无 .env 文件时静默跳过（不影响以纯环境变量启动）。"""
        app_module._load_dotenv(str(tmp_path / "not-exist.env"))
