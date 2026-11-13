# tests/test_detector.py
"""YOLOv8 ONNX 推理骨架测试：MODEL_PATH 守卫 + 合成模型端到端。

合成模型输出为常量 (1, 10, 8400)：4 列框坐标（cx,cy,w,h，640 坐标系）
+ 6 列类别分数（训练顺序 BIODEGRADABLE/CARDBOARD/GLASS/METAL/PAPER/PLASTIC），
用于离线验证解码、阈值过滤、NMS、类别映射与坐标换算全链路。
"""

import cv2
import numpy as np
import pytest

from services.detector import (
    CLASS_MAP,
    DEFAULT_CLASS_NAMES,
    ImageDecodeError,
    ModelNotReadyError,
    YoloDetector,
)

pytest.importorskip("onnx", reason="合成测试模型需要 onnx 包")


def _build_fake_onnx(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    arr = np.zeros((1, 10, 8400), dtype=np.float32)

    def put(anchor, cx, cy, w, h, cls, score):
        arr[0, 0, anchor] = cx
        arr[0, 1, anchor] = cy
        arr[0, 2, anchor] = w
        arr[0, 3, anchor] = h
        arr[0, 4 + cls, anchor] = score

    put(100, 320, 320, 100, 200, 0, 0.90)  # BIODEGRADABLE -> trash（保留）
    put(101, 322, 321, 101, 201, 5, 0.85)  # 重叠 PLASTIC -> 应被 NMS 抑制
    put(200, 100, 100, 50, 50, 4, 0.10)  # PAPER 低置信 -> 应被阈值过滤
    put(300, 630, 630, 60, 60, 3, 0.77)  # METAL 越界 -> 应被裁剪回原图

    const = helper.make_node(
        "Constant", [], ["output"], value=numpy_helper.from_array(arr, name="const_out")
    )
    graph = helper.make_graph(
        [const],
        "fake6",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 8400])],
    )
    onnx.save(helper.make_model(graph, producer_name="tests"), path)
    return str(path)


@pytest.fixture(scope="module")
def fake_model(tmp_path_factory):
    return _build_fake_onnx(tmp_path_factory.mktemp("models") / "fake.onnx")


@pytest.fixture(scope="module")
def png_640x480_bytes():
    image = np.random.RandomState(42).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------ MODEL_PATH 守卫


def test_no_model_path_raises(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(ModelNotReadyError):
        YoloDetector()


def test_nonexistent_model_file_raises(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "Z:/definitely/not/here.onnx")
    with pytest.raises(ModelNotReadyError):
        YoloDetector()


def test_invalid_onnx_file_raises(monkeypatch, tmp_path):
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not an onnx model at all")
    monkeypatch.setenv("MODEL_PATH", str(broken))
    with pytest.raises(ModelNotReadyError):
        YoloDetector()


def test_class_map_matches_contract_labels():
    assert set(CLASS_MAP.values()) == {
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    }
    assert CLASS_MAP["BIODEGRADABLE"] == "trash"


# ------------------------------------------------------------------ 端到端推理


def test_detect_end_to_end(fake_model, png_640x480_bytes):
    detector = YoloDetector(model_path=fake_model)
    # 合成模型无 names 元数据 -> 回退训练顺序默认表
    assert detector.class_names == DEFAULT_CLASS_NAMES

    detections, width, height = detector.detect(png_640x480_bytes)
    assert (width, height) == (640, 480)

    # 期望：trash(0.90) 与 metal(0.77)；重叠 PLASTIC 被 NMS 抑制、低置信 PAPER 被过滤
    assert [d["class"] for d in detections] == ["trash", "metal"]
    assert detections[0]["confidence"] == 0.90
    assert detections[1]["confidence"] == 0.77

    # scale_x=1.0, scale_y=0.75：trash 框 (270,220,370,420)->(270,165,370,315)
    assert detections[0]["box"] == [270, 165, 370, 315]
    # metal 框 640 空间 (600,600,660,660) -> (600,450,495*0.75=495->裁到 480)
    assert detections[1]["box"] == [600, 450, 640, 480]

    # 每条严格符合契约形状
    for det in detections:
        assert set(det) == {"class", "confidence", "box"}
        assert 0.0 <= det["confidence"] <= 1.0
        x1, y1, x2, y2 = det["box"]
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height


def test_decode_failure_raises(fake_model):
    detector = YoloDetector(model_path=fake_model)
    with pytest.raises(ImageDecodeError):
        detector.detect(b"\x00\x01\x02 garbage bytes")


def test_model_path_env_used_when_no_argument(fake_model, monkeypatch, png_640x480_bytes):
    """不传参数时应从 MODEL_PATH 环境变量加载（生产挂载方式）。"""
    monkeypatch.setenv("MODEL_PATH", fake_model)
    detector = YoloDetector()
    detections, _, _ = detector.detect(png_640x480_bytes)
    assert detections  # 合成模型必有两个可检出目标
