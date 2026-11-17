# V3 接入检查日志（2026-11-17）

| 检查 | 结果 |
| --- | --- |
| `python RTDETR_DUT_ResearchV3/install_v3.py --repo .` | Dry Run 通过，计划新增 8 文件、补丁 `tasks.py` |
| 安装器 `--apply` | 完成；原始 `rtdetr-r18.yaml` 未修改 |
| `python -m compileall -q ...` | 通过 |
| 六份 YAML 与原始 R18 `head` 对照 | 均完全相同，保持 P3/P4/P5 与 300 Query |
| `python RTDETR_DUT_ResearchV3/run_v3.py --repo . --exp gap --device 0` | Dry Run 通过，显示 150 epoch / batch8 / AdamW / lr0=1e-4 |
| `python -m pytest -q RTDETR_DUT_ResearchV3/tests/test_signals.py` | 收集失败：本机 `ModuleNotFoundError: No module named 'torch'` |
| 真实模型 FP32、AMP、空/多 GT、DN、Backward、梯度、EMA/Checkpoint、Val | 未运行；需在有 PyTorch 的服务器运行 `python -m pytest -q RTDETR_DUT_ResearchV3/tests/test_signals.py RTDETR_DUT_ResearchV3/tests/test_integration.py` |

安装器原样输出的训练期 Hook 使用 `try/finally` 在前向后移除。是否在真实工程的异常、二次前向及 EMA 深拷贝中完全安全，需服务器集成测试确认。V3 四项训练结果均未运行。
