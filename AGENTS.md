# AGENTS.md

## 项目：智能垃圾分类识别与投放指导系统

### 项目目标
基于 YOLO 视觉检测 + DeepSeek 大模型智能体的 Web 应用：
用户上传图片 → YOLO 输出检测框与细粒度类别 → 智能体调用本地工具
（类别映射、投放规则查询）生成四分类投放建议 → 网页展示完整结果。

### 硬性约束（不得违反）
- 接口契约以 `docs/api.md` 为唯一事实来源，字段名、类型、错误码不得擅自修改。
- 修改任何接口行为前，必须先更新 `docs/api.md`，再改代码。
- API Key 只从环境变量 `DEEPSEEK_API_KEY` 读取；禁止硬编码、打印、写入日志或提交。
- 模型文件（`*.pt`、`*.onnx`）不提交 git，通过环境变量 `MODEL_PATH` 指定的路径加载。
- 必须保留 mock 模式：`MOCK=1` 时在无模型、无网络条件下可完整启动并演示流程。
- 不得删除或绕过 `tests/` 中的测试；测试失败不得提交。

### 目录结构与职责
- `app.py`：Flask 入口，只做路由、参数校验、错误响应，业务逻辑放 `services/`
- `services/detector.py`：YOLO ONNX 推理，输入图片 bytes，输出检测列表
- `services/advisor.py`：DeepSeek 工具调用循环，输入检测列表，输出建议
- `services/rules.py`：类别映射表与投放规则查询（供智能体调用的工具函数）
- `templates/`、`static/`：前端页面与静态资源
- `data/rules.json`：投放规则库，不进镜像，通过挂载或环境变量指定路径
- `tests/`：接口与核心逻辑测试
- `train/`：训练脚本与数据配置，与运行环境隔离
- `docs/api.md`：接口契约

### 训练与推理分离
- 运行环境只安装 `requirements.txt`，不得引入 `torch`、`ultralytics`。
- 训练代码在 `train/`，依赖 `requirements-train.txt`，在宿主机或 Colab 运行。
- 模型产物 `best.onnx` 通过 `MODEL_PATH` 挂载，不提交到 git，不写入 Docker 镜像。

### 命令
- 安装运行依赖：`pip install -r requirements.txt`
- mock 启动：`MOCK=1 python app.py`
- 运行测试：`pytest tests/`
- 构建镜像：`docker build -t garbage-agent .`
- 启动容器：`docker run -p 5000:5000 -e DEEPSEEK_API_KEY=... -e MODEL_PATH=/models/best.onnx -v ./models:/models garbage-agent`

### 代码来源与修改原则
- 本项目骨架代码（Flask 路由、前端页面、mock 实现、Dockerfile 初版）
  由 DeepSeek 网页版生成，经本人核验后提交。生成记录与核验方式见报告附录 2。
- 本地 agent 的职责是：发现并修复真实缺陷、删除明显冗余、补全真实实现；
  不是重写风格、调整架构或改变接口。
- 修改前先读 `docs/api.md`；任何修改不得改变契约中的字段名、类型和错误码。
- 修改后必须运行 `pytest tests/`，并在提交信息中说明"修了什么、为什么"。

### 冗余处理
允许处理：
- 未使用的 import、变量、函数
- mock 分支中已被真实实现完全替代的死代码
- 明显重复的代码块（可抽成函数，但不得改变对外行为）

不允许：
- 仅因风格不同而重写已工作的代码
- 删除注释、日志和错误处理
- 合并或拆分现有接口
- 引入新依赖来"简化"代码

### 修改规则
- 每次修改后必须运行 `pytest tests/`，失败则不得提交。
- 不新增依赖，除非在 `requirements.txt` 中锁定版本并说明原因。
- 只改与当前任务相关的文件，不做顺手重构。
- 不修改 `.gitignore` 中已忽略的规则，除非任务明确要求。

### 当前阶段优先级（前端优先）
1. 前端页面与交互（上传、检测框渲染、建议展示）
2. mock 接口链路跑通（`MOCK=1` 下可完整演示）
3. 真实 YOLO ONNX 推理接入
4. DeepSeek 智能体工具调用接入
5. Docker 封装与联调

存在多个待办时按此顺序处理；前端未跑通前，不启动后端真实实现。
后端未就绪时，前端一律使用 mock 数据，不得因等待后端而阻塞。

### 完成标准（每个阶段自检）
- 接口：请求/响应与 `docs/api.md` 一致，错误码覆盖正常与异常。
- 前端：浏览器可完成"上传 → 看到结果"的完整操作，无需命令行。
- 后端：`MOCK=0` 且模型与 API 就绪时可切换真实实现，失败时能降级到明确错误信息。
- 提交：`git log` 中每次提交可追溯到具体功能或修复。
