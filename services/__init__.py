# services/__init__.py
"""业务服务包：检测（detector）、建议（advisor）、规则（rules）。

app.py 只负责路由与参数校验，业务逻辑一律放在本包内。
本包的依赖（onnxruntime / opencv 等）仅在非 mock 模式下按需导入，
保证 MOCK=1 时无模型、无相关依赖也能完整启动。
"""
