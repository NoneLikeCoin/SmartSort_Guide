# services/detector.py
"""YOLOv8 ONNX 推理服务。

输入：图片二进制 bytes；输出：符合接口契约的检测列表 + 原图尺寸。
契约要点（docs/api.md 通用约定）：
  - confidence 为 0—1 浮点数；
  - box 为原图像素坐标 [x1, y1, x2, y2]；
  - class 只允许契约的 6 个标签（cardboard/glass/metal/paper/plastic/trash）。

本模块依赖 onnxruntime / opencv / numpy，仅在非 mock 模式下被 app.py 按需导入。
"""

import ast
import os

import cv2
import numpy as np
import onnxruntime as ort

# ------------------------------------------------------------------ 推理常量

# YOLOv8 标准输入尺寸（640x640）
INPUT_SIZE = 640

# 置信度过滤阈值与 NMS IoU 阈值
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ------------------------------------------------------------------ 类别映射

# 训练类别名 -> 契约类别标签。
# 训练集没有独立的 TRASH 类：BIODEGRADABLE（可降解物）归入其他垃圾 trash。
CLASS_MAP = {
    "BIODEGRADABLE": "trash",
    "CARDBOARD": "cardboard",
    "GLASS": "glass",
    "METAL": "metal",
    "PAPER": "paper",
    "PLASTIC": "plastic",
}

# 读不到 ONNX 元数据时的兜底类别名顺序，
# 必须与训练配置（train/ 数据集 data.yaml）的类别顺序完全一致。
DEFAULT_CLASS_NAMES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC",
]

# ------------------------------------------------------------------ 异常定义


class ModelNotReadyError(Exception):
    """模型尚未就绪：MODEL_PATH 未设置、文件不存在或 ONNX 加载失败。"""


class ImageDecodeError(Exception):
    """上传的图片无法解码。"""


# ------------------------------------------------------------------ 检测器


class YoloDetector:
    """加载 YOLOv8 ONNX 模型并执行检测。

    用法：
        detector = YoloDetector()                 # 从 MODEL_PATH 环境变量加载
        detections, width, height = detector.detect(image_bytes)

    detect() 的第一个返回值是严格符合契约的列表：
        [{"class": "trash", "confidence": 0.8, "box": [x1, y1, x2, y2]}]
    第二个、第三个返回值是原图宽高（像素），供 /detect 填充 image_size 字段；
    随列表一并返回是为了避免上层对同一份 bytes 重复解码。
    """

    def __init__(self, model_path=None):
        path = (model_path or os.environ.get("MODEL_PATH", "")).strip()
        if not path:
            raise ModelNotReadyError("未设置 MODEL_PATH 环境变量。")
        if not os.path.isfile(path):
            raise ModelNotReadyError(
                "MODEL_PATH 指向的模型文件不存在：{}".format(path)
            )
        try:
            self.session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime 抛出的异常类型繁多，统一收敛
            raise ModelNotReadyError("ONNX 模型加载失败：{}".format(exc)) from exc

        self.input_name = self.session.get_inputs()[0].name
        self.class_names = self._load_class_names()

    # ---------------------------------------------------------- 模型元数据

    def _load_class_names(self):
        """优先从 ONNX 元数据读取类别名（ultralytics 导出时写入 names 字段），
        读不到则回退到 DEFAULT_CLASS_NAMES。
        """
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            raw = meta.get("names")
            if raw:
                # ultralytics 以 Python dict 字面量存储，如 "{0: 'PAPER', ...}"
                names_dict = ast.literal_eval(raw)
                return [str(names_dict[i]) for i in range(len(names_dict))]
        except Exception:
            # 元数据缺失或格式异常不致命：回退到训练顺序的固定列表
            pass
        return list(DEFAULT_CLASS_NAMES)

    # ---------------------------------------------------------- 对外接口

    def detect(self, image_bytes):
        """检测一张图片，返回 (契约检测列表, 原图宽, 原图高)。"""
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ImageDecodeError("无法解码上传的图片。")

        height, width = image.shape[:2]
        blob = self._preprocess(image)
        output = self.session.run(None, {self.input_name: blob})[0]
        detections = self._postprocess(output, width, height)
        return detections, width, height

    # ---------------------------------------------------------- 预处理

    @staticmethod
    def _preprocess(image):
        """BGR 原图 -> (1, 3, 640, 640) float32 RGB，像素值归一化到 0—1。"""
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = blob[np.newaxis, ...]  # 加 Batch 维度
        return np.ascontiguousarray(blob)

    # ---------------------------------------------------------- 后处理

    def _postprocess(self, output, width, height):
        """解析 YOLOv8 标准导出输出 (nc+4, 8400)：
        阈值过滤 -> NMS -> 类别映射 -> 坐标换算回原图 xyxy。

        注：仅支持标准（非 end2end）导出格式；用 end2end=True 导出的
        模型需另行适配。
        """
        data = np.squeeze(output).astype(np.float32)
        if data.ndim != 2 or data.shape[1] < 5:
            raise RuntimeError(
                "模型输出形状不符合 YOLOv8 预期：{}".format(output.shape)
            )

        # 标准导出为 (nc+4, 8400)，转置成 (8400, nc+4) 便于按锚点行处理
        if data.shape[0] < data.shape[1]:
            data = data.T

        boxes_xywh = data[:, :4]  # cx, cy, w, h（640 坐标系）
        scores = data[:, 4:]  # (anchors, nc)
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences >= CONF_THRESHOLD
        cand_boxes = boxes_xywh[keep]
        cand_ids = class_ids[keep]
        cand_confs = confidences[keep]

        if cand_boxes.shape[0] == 0:
            return []

        # 640 坐标系下 cx,cy,w,h -> 左上角 x,y,w,h，供 cv2.dnn.NMSBoxes 使用
        nms_boxes = []
        for cx, cy, bw, bh in cand_boxes:
            nms_boxes.append(
                [float(cx - bw / 2.0), float(cy - bh / 2.0), float(bw), float(bh)]
            )
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, cand_confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD
        )
        if indices is None or len(indices) == 0:
            return []
        # 不同 OpenCV 版本返回 (n,) 或 (n,1)，统一展平
        kept = np.asarray(indices, dtype=int).ravel()

        scale_x = width / float(INPUT_SIZE)
        scale_y = height / float(INPUT_SIZE)

        detections = []
        for idx in kept:
            cls_id = int(cand_ids[idx])
            if cls_id >= len(self.class_names):
                continue  # 输出通道与类别名数量不一致，跳过该框
            raw_name = self.class_names[cls_id]
            label = CLASS_MAP.get(raw_name)
            if label is None:
                # 不属于本项目 6 类（如误用 COCO 预训练权重），
                # 直接丢弃，避免向契约注入非法 class 值。
                continue

            cx, cy, bw, bh = cand_boxes[idx]
            x1 = (cx - bw / 2.0) * scale_x
            y1 = (cy - bh / 2.0) * scale_y
            x2 = (cx + bw / 2.0) * scale_x
            y2 = (cy + bh / 2.0) * scale_y

            # 裁剪回原图边界；坐标为原图像素，取整与 mock 数据保持一致
            x1 = int(max(0.0, min(x1, width)))
            y1 = int(max(0.0, min(y1, height)))
            x2 = int(max(0.0, min(x2, width)))
            y2 = int(max(0.0, min(y2, height)))

            detections.append(
                {
                    "class": label,
                    "confidence": round(float(cand_confs[idx]), 2),
                    "box": [x1, y1, x2, y2],
                }
            )

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections
