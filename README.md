# 智能垃圾分类识别与投放指导系统

基于 **YOLOv8 视觉检测 + DeepSeek 大模型智能体** 的 Web 应用：
上传图片 → YOLO 输出检测框与细粒度类别 → 智能体调用本地工具（类别映射、投放规则查询）
生成四分类投放建议 → 网页展示检测框与完整建议。

接口契约以 [`docs/api.md`](docs/api.md) 为唯一事实来源；开发约束见 [`AGENTS.md`](AGENTS.md)。

## 目录结构

```
app.py                  Flask 入口：路由、参数校验、统一错误响应
services/
  detector.py           YOLOv8 ONNX 推理（预处理 / NMS / 类别映射 / 坐标换算）
  advisor.py            DeepSeek 智能体：多轮工具调用循环与结构化输出校验
  rules.py              类别映射表与投放规则查询（供智能体调用的本地工具）
data/rules.json         投放规则库（可用 RULES_PATH 指向挂载副本）
templates/ static/      前端页面与资源
docs/api.md             接口契约
tests/                  接口 / 智能体 / 推理的离线测试（pytest）
train/                  训练脚本与数据集（与运行环境隔离，产物不进仓库）
scripts/git-hooks/      pre-commit 测试门禁
```

## 环境要求

- Python 3.12+，安装运行依赖（**不含 torch/ultralytics**，见 AGENTS.md 训练/推理分离）：

```bash
pip install -r requirements.txt
```

## 快速启动（mock 模式，推荐先跑这个）

无需模型、无需网络、无需密钥，即可完整演示 上传 → 检测框 → 投放建议：

```bash
MOCK=1 python app.py
# 浏览器打开 http://localhost:5000
```

PowerShell 下用 `$env:MOCK="1"; python app.py`。

## 真实运行

两个开关各自独立，可分步接入：

```bash
# 1) 视觉检测：指向 YOLOv8 导出的 ONNX（训练产物 best.onnx，不提交 git）
set MODEL_PATH=D:/models/best.onnx
set MOCK=0
python app.py            # 模型缺失时 /detect 返回 503 MODEL_NOT_READY，前端可感知

# 2) 投放建议：DeepSeek 智能体（工具调用 map_category / query_rule）
set DEEPSEEK_API_KEY=sk-xxxx
python app.py            # 密钥缺失时 /advise 返回 503 LLM_UNAVAILABLE
```

也可以把上述变量写进项目根目录的 `.env`（参考 `.env.example`）：
app 启动时自动加载，**系统环境变量优先于 .env**；`.env` 已被 gitignore，密钥绝不入库。

`GET /health` 在真实模式下如实报告 `model_loaded / llm_configured` 就绪状态。

模型导出（训练完成后，在装有 ultralytics 的训练环境执行）：

```bash
yolo export model=runs/detect/<run>/weights/best.pt format=onnx imgsz=640
```

## 测试与提交门禁

```bash
python -m pytest tests/          # 34 个用例，全部离线（合成 ONNX + 伪造 API 客户端）
git config core.hooksPath scripts/git-hooks   # 每个克隆执行一次：启用"测试不过不得提交"
```

## Docker（阶段 5，进行中）

```bash
docker build -t garbage-agent .
docker run -p 5000:5000 -e DEEPSEEK_API_KEY=... -e MODEL_PATH=/models/best.onnx -v ./models:/models garbage-agent
```

Dockerfile 与 `.dockerignore`（排除 `data/`、`train/`、`*.onnx` 等）尚未生成，将在联调阶段交付。
