# API 契约

本文件是前后端、模型、智能体之间的唯一接口约定。
修改任何字段名、类型或错误码前，必须先更新本文件，再改代码。

## 通用约定

- Base URL：`http://localhost:5000`
- 所有响应均为 `application/json; charset=utf-8`
- 时间单位：毫秒；置信度：0—1 浮点数；坐标：原图像素坐标
- 错误响应统一格式：

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "无法解码上传的图片"
  }
}
