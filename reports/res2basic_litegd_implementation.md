# RT-DETR-R18 Res2-BasicBlock 与 Lite-GDNeck 实现记录

## 范围与结构

原始 `rtdetr-r18.yaml`、`BasicBlock`、CCFM 和 Decoder 未修改。B1 仅将 Stage3 第二个 BasicBlock 的第一处 3×3 分支替换为 `Res2Conv`，第二处 3×3、Shortcut、相加和激活沿用原结构。scale=4、每组宽度 32；B5 关闭层级连接，保留同样分组宽度。层级连接参考 [Res2Net 官方实现](https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net_v1b.py)，这里适配 BasicBlock，不是复刻其 Bottle2neck。

Lite-GDNeck 取 P3(128)、P4(256) 和原有 input_proj+AIFI 后的 P5(256)，先在 P5 尺度 Gather 并向 P4/P5 分配，再在 P4 尺度 Gather 并向 P3/P4/P5 分配。各尺度有独立可学习、有界的残差增益，最终输出三个 256 通道 Tensor，经 `GDFeature` 路由给原 Decoder（300 queries）。这是受 [Gold-YOLO 官方 GD 代码](https://github.com/huawei-noah/Efficient-Computing/blob/master/Detection/Gold-YOLO/gold_yolo/reppan.py)启发的轻量实现；这里的高低层顺序、卷积融合和通道数按本项目另行设计。

## 新文件

- `ultralytics/nn/modules/res2basic.py`：Res2Conv、Res2BasicBlock、Res2Blocks。
- `ultralytics/nn/modules/lite_gd.py`：高低尺度 Gather、独立 Injection、Neck、输出路由。
- `tools/generate_res2_gd_yaml.py`：从未修改的 R18 配置生成 11 个独立 YAML。
- `tools/train_r18_res2_gd.py`、`evaluate_r18_res2_gd.py`、`benchmark_r18_res2_gd.py`、`diagnose_r18_res2_gd.py`：服务器执行入口。
- `tests/test_res2basic_litegd.py`：独立结构、梯度、配置及服务器整网测试。

`ultralytics/nn/tasks.py` 注册多输入和多输出模块。多输出使用 tuple 通道元数据和三个显式 Selector，以兼容本仓库 `parse_model` 的保存列表和 Decoder 接口。

## 本地验证结果

- 独立测试 6 项中 4 项通过；CUDA AMP 项因本机无 CUDA 跳过，完整 Ultralytics 模型项因本机缺少 `cv2` 跳过。
- 新增 Python 文件编译通过；`git diff --check` 通过。
- 单独测得 128 通道原 BasicBlock 为 295,424 参数，替换后的 Res2BasicBlock 为 208,832 参数；LiteGDNeck 为 987,013 参数。这些不是完整模型参数量。
- LiteGDNeck 单独在 640 输入对应的三层特征上，由 PyTorch profiler 支持的算子计得约 4.479 GFLOPs。此值是受支持算子的下界，不能直接作为完整模型 GFLOPs。
- `--dry-run` 验证训练设置为 150 epoch、640、微批量 2、累积 4、有效批量 8、AdamW、初始 LR 0.0001、FP32。

服务器部署后先运行 `python -m unittest discover -s tests -p test_res2basic_litegd.py -v`。训练入口也会自动运行这组测试、构建整网并检查 Decoder 输入形状；任一失败即停止。完整 640 前向、反向、AMP 和加载保存路径仍待服务器实测，不应将本地跳过视为通过。

## 训练协议限制

本地未找到原始 R18 的 `args.yaml`、`results.csv` 和 checkpoint，无法核实历史 weight decay、增强、验证频率及真实 LR 曲线。训练入口支持 `--baseline-args /path/to/args.yaml` 做核心设置比对；没有该文件时会发出明确警告，并使用仓库现有 `anti_detr_default.yaml`。本项目 Trainer 把 `warmup_epochs` 当成**微批量迭代次数**；微批量 2、累积 4 时，2000 次优化器迭代对应 8000 次微批量迭代。实际每 epoch LR 和 loss 由 Trainer 的 `results.csv` 写出，训练完成后复制到 `epoch_history.json`。历史协议若不同，需要同协议 Baseline，不能直接用旧分数作严格公平对照。

尚未执行任何 150 epoch 训练或生成检测 AP。目标指标须由服务器实际评估产生。
