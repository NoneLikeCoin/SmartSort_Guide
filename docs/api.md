# API 契约

本文件是前后端、模型、智能体之间的唯一接口约定。修改任何字段名、类型或错误码前，必须先更新本文件，再改代码。

## 通用约定
- Base URL：http://localhost:5000
- 所有响应均为 application/json; charset=utf-8
- 时间单位：毫秒；置信度：0—1 浮点数；坐标：原图像素坐标
- 错误响应统一格式：

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "无法解码上传的图片"
  }
}
```

## 1. POST /detect
对上传图片执行 YOLO 目标检测，返回检测框与细粒度类别。

### 请求
- Content-Type: multipart/form-data
- 字段：`image` (file, 必填, jpg/png, ≤10MB)

### 响应 200

```json
{
  "request_id": "a1b2c3",
  "detections": [
    {"class": "plastic", "confidence": 0.87, "box": [120, 80, 340, 410]}
  ],
  "image_size": {"width": 800, "height": 600},
  "elapsed_ms": 186
}
```

- `detections[].class`: string, 仅限第 5 节列出的标签
- `detections[].box`: [x1, y1, x2, y2] 原图像素坐标
- 未检测到目标时 `detections: []`，HTTP 200

## 2. POST /advise
接收 /detect 的检测结果，由智能体生成四分类投放建议。

### 请求

```json
{
  "detections": [{"class": "plastic", "confidence": 0.87}],
  "city": "changsha",
  "scene": "campus"
}
```

### 响应 200

```json
{
  "request_id": "d4e5f6",
  "items": [
    {
      "class": "plastic",
      "category": "可回收物",
      "bin": "蓝色可回收物桶",
      "reason": "塑料瓶属于可回收物，投放前请清空内容物并压扁。"
    }
  ],
  "summary": "本次检测到 1 件物品……",
  "elapsed_ms": 2310
}
```

## 3. GET /health

```json
{"status":"ok","mock":false,"model_loaded":true,"llm_configured":true}
```

## 4. mock 模式
环境变量 `MOCK=1` 时，返回固定 mock 数据。`MOCK=0` 时走真实实现。

## 5. 类别标签
cardboard=纸板, glass=玻璃, metal=金属, paper=纸, plastic=塑料, trash=其他垃圾
