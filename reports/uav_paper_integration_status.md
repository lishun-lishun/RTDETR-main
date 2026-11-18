# UAV Paper Innovations 接入状态（2026-11-18）

## 已完成

- 在当前 `codex/uav-research-integration` 分支接入四个模块和七份独立 YAML。原始 `rtdetr-r18.yaml`、权重、历史 `runs/` 未改动。
- 修复包内安装器：默认 Dry Run、显式 `--apply`；匹配本分支的 `parse_model()` Decoder 分支，缺少锚点时明确报错。连续执行两次 `--apply`，第二次未重复插入。
- 修复训练脚本的实验协议：batch=8、micro-batch=8、150 epochs、AdamW、lr0=0.0001、lrf=1、warmup_epochs=2000（本 Trainer 按微批迭代数解释）、AMP=false、mosaic=0、pretrained=false、seed=0；默认 Dry Run，必须显式 `--execute`。
- 四模块代码已安装于 `ultralytics/nn/uav_innovations/`，`tasks.py` 和 `modules/__init__.py` 的补丁通过 `git diff --check`；Python 静态编译通过。

## 尚未验证

本机 Python 没有 `torch`，所以真实 YAML 构建、张量 shape、FP32/AMP、Batch1/2、0/1/多 GT、DN、Forward/Backward、梯度有限性、Checkpoint/EMA 保存加载均**未运行**。包内 `integration/test_integration.py` 主要使用自建的模拟 RT-DETR 图；它不能替代在本仓库真实 `RTDETRDetectionModel` 上的测试。包内参数量/GFLOPs 与论文新颖性主张也未被本机实测证实。

## 服务器先执行

```bash
python -m pytest -q RTDETR_UAV_Paper_Innovations/innovation_1/test_module.py RTDETR_UAV_Paper_Innovations/innovation_2/test_module.py RTDETR_UAV_Paper_Innovations/innovation_3/test_module.py RTDETR_UAV_Paper_Innovations/innovation_4/test_module.py
python -m pytest -q RTDETR_UAV_Paper_Innovations/integration/test_integration.py
python tools/smoke_uav_paper_real.py --device cuda:0 --size 128
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E1 --epochs 150 --batch 8 --micro-batch 8 --seed 0 --device 0
```

前两条是包内测试；第三条使用真实仓库模型执行构建、推理、0/1/多 GT、FP32/AMP、反向、梯度与完整模型/EMA 序列化；第四条只打印训练配置。服务器运行第三条通过后，再考虑 `--execute`。新旧模块不能凭包内 Mock 测试就宣称兼容。

## 当前最重要的工程风险

- `SAHNWDRTDETRLoss` 是独立 `nn.Module` 实现，不继承仓库原 `RTDETRDetectionLoss`；需要服务器直接核查与原 Matcher、VFL、辅助层、DN 的等价性。
- `UADCRDecoder` 独立实现了 Decoder，而不是继承原类；原始权重的 `state_dict` 键兼容性需显式比较。
- 包内 README 的目录和训练协议描述与实际文件及历史配置有不一致处；以本报告和修订后的训练脚本为准。
- 七份新 YAML 的解析和推理张量 shape 需要真实 PyTorch 环境验证。没有这些结果时，不能提交论文性能或“零开销”结论。
