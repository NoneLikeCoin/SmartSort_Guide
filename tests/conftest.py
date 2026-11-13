# tests/conftest.py
"""测试公共夹具：路径注入、环境变量隔离、Flask test client、有效 PNG 字节。"""

import os
import sys

import pytest

# 使测试可直接 import app / services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """每个用例前清空相关环境变量，并复位检测器全局缓存，保证用例互不影响。"""
    for var in ("MOCK", "MODEL_PATH", "DEEPSEEK_API_KEY", "RULES_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_module, "_detector", None)
    yield
    monkeypatch.setattr(app_module, "_detector", None)


@pytest.fixture
def client():
    """Flask 测试客户端（不占端口、不起服务）。"""
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_png_bytes():
    """8x8 纯色 PNG 的原始字节（会话级共享，头部魔数合法、可解码）。"""
    import cv2
    import numpy as np

    image = np.random.RandomState(0).randint(0, 255, (8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()
