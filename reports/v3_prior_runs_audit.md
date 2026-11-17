# V3 历史运行审计

来源：`runs/uav_ablation/*/seed0/results.csv`、`args.yaml`、`training_config.json`。程序复核输出见 [v3_prior_runs.csv](v3_prior_runs.csv)。这些是训练期间验证集指标，不是此前历史测试集 AP50:95=0.707。

- E0 有 124 轮；E1–E8 有 150 轮；TinyV2 M1–M4 分别有 123、120、70、68 轮。未完成模型不能与完整 150 轮的峰值直接比较。
- 第 120 轮，TinyV2 M2 对 E0 差值为 -0.00697；E4 为 +0.00128；E1 为 -0.00208；E5 为 -0.00529；E8 为 -0.00570。
- 第 123 轮，TinyV2 M1 对 E0 为 +0.00103，只有 seed0。第 70 轮 M3 为 -0.00139，第 68 轮 M4 为 -0.00123；后两者尚未完成。
- `training_config.json` 使用 `pretrained=false`、AdamW、`lr0=0.0001`、`lrf=1`、batch/micro-batch=8、imgsz=640、AMP=false。`ultralytics/engine/trainer.py` 第 298 行直接令 `nw=self.args.warmup_epochs`，所以 2000 的单位是微批迭代，不是 2000 个 epoch。
- E1–E8 的代码提交为 `5607e4a`，E0/TinyV2 为 `fdd6c25`。两提交间的基线相关差异仅在 `tasks.py` 添加 TinyV2 可选分支及解析器识别，原始 R18 Decoder、Matcher、Loss 和 Trainer 未变；仍应以完整 150 轮 E0 作最终同协议对照。
- `dataset/DUT-Anti-UAV.yaml` 的 train、val、test 分别指向 `images/train`、`images/val`、`images/test`。本地只有训练日志，尚未完成图片级跨划分去重、GT 尺寸统计、误检/漏检分类、AP75/85/90 和 Encoder Top300 覆盖率。不能由现有 CSV 推断这些量。

复算命令：`python RTDETR_DUT_ResearchV3/analyze_uploaded_runs.py --repo . --out reports/v3_prior_runs.csv`。V3 单模块和组合实验均未运行。
