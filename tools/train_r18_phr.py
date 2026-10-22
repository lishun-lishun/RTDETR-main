#!/usr/bin/env python3
"""RT-DETR-R18 PHR (Partial High-Resolution Residual Backbone) 训练脚本 (Linux 服务器端).

在 DUT-Anti-UAV 数据集上训练 PHR-ResNet18 的三个递进版本 PHR-1 / PHR-2 / PHR-3。
默认只打印训练计划 (dry-run), 需要显式 ``--train`` 才真正启动, 避免误触发长训练;
已存在的同名运行目录默认跳过 (用 ``--force`` 覆盖提示)。

重要约定
--------
* 本任务只研究 Backbone, 不修改 Neck / Decoder, 不重训 Baseline。
* 按规格书 **只训练 PHR-1**; PHR-2 / PHR-3 仅完成代码 + YAML 验证 (见
  ``tools/validate_phr.py``), 本脚本保留其入口以支持后续复现, 但默认不纳入。
* 训练参数从 ``ultralytics/cfg/anti_detr_default.yaml`` 读取 (与 Baseline/BAPS/RDCF/
  LSCE/ADCF 完全一致), warmup_epochs 字段沿用本项目约定 (表示 warmup iterations),
  并按 ``batch // micro_batch`` 累加步数。

用法示例
--------
    # 查看计划 (不训练):
    python tools/train_r18_phr.py --model phr1
    python tools/train_r18_phr.py --model all --device 0,1,2

    # 单模型训练 (独立实验目录, 与 Baseline/BAPS/RDCF/LSCE/ADCF 互不干扰):
    python tools/train_r18_phr.py --model phr1 --device 0 --train
    python tools/train_r18_phr.py --model phr2 --device 1 --train   # 按规格书通常不跑
    python tools/train_r18_phr.py --model phr3 --device 0,1,2 --train  # 按规格书通常不跑

    # 从检查点恢复 / 从权重初始化:
    python tools/train_r18_phr.py --model phr1 --resume --device 0
    python tools/train_r18_phr.py --model phr1 --weights runs/r18_phr1/train/.../best.pt --train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RTDETR_MODEL_DIR = _PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DATA_CONFIG = _PROJECT_ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE_CONFIG = _PROJECT_ROOT / "ultralytics" / "cfg" / "anti_detr_default.yaml"

# 模型角色 -> (YAML 文件名, 独立项目目录名)
_MODELS = {
    "phr1": ("rtdetr_r18_phr1.yaml", "r18_phr1"),
    "phr2": ("rtdetr_r18_phr2.yaml", "r18_phr2"),
    "phr3": ("rtdetr_r18_phr3.yaml", "r18_phr3"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="+",
        choices=("all", *_MODELS),
        default=("phr1",),
        help="模型角色; 'all' 依次覆盖全部三种配置 (通常只需 phr1)。",
    )
    parser.add_argument("--device", default=None, help="例如: 0, 0,1, 或 cpu。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练 epochs。")
    parser.add_argument("--batch", type=int, default=None, help="覆盖全局 batch。")
    parser.add_argument("--micro-batch", type=int, default=None, help="覆盖 micro-batch。")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖输入尺寸。")
    parser.add_argument("--workers", type=int, default=None, help="覆盖数据加载 workers。")
    parser.add_argument("--lr0", type=float, default=None, help="覆盖初始学习率。")
    parser.add_argument("--weights", type=Path, default=None, help="从检查点初始化 (仅单模型)。")
    parser.add_argument("--resume", action="store_true", help="从 latest.pt 恢复 (仅单模型)。")
    parser.add_argument("--train", action="store_true", help="真正启动训练; 默认只打印计划。")
    parser.add_argument("--force", action="store_true", help="允许在已有运行目录上继续 (覆盖提示)。")
    parser.add_argument(
        "--project",
        type=Path,
        default=_PROJECT_ROOT / "runs",
        help="顶层项目目录, 默认 runs/。",
    )
    return parser.parse_args()


def selected(names):
    if "all" in names:
        return dict(_MODELS)
    return {n: _MODELS[n] for n in names}


def load_profile(args):
    profile = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
    for key in ("epochs", "batch", "imgsz", "workers", "lr0"):
        value = getattr(args, key)
        if value is not None:
            profile[key] = value
    if args.micro_batch is not None:
        profile["micro_batch"] = args.micro_batch
    if args.device is not None:
        profile["device"] = args.device
    return profile


def device_count(device):
    if device is None or not isinstance(device, str):
        return 1
    devices = [d.strip() for d in device.split(",") if d.strip()]
    return len(devices) if all(d.isdigit() for d in devices) else 1


def training_kwargs(profile):
    """把 paper batch 翻译成内存安全的 optimizer 更新 (与 train_dut_anti_uav.py 一致)。"""
    settings = {k: v for k, v in profile.items() if k not in ("test_batch", "micro_batch")}
    steps = profile["batch"] // profile["micro_batch"]
    settings["batch"] = profile["micro_batch"]
    settings["gradient_accumulation"] = steps
    settings["warmup_epochs"] = profile["warmup_epochs"] * steps
    return settings


def validate_profile(profile):
    world = device_count(profile.get("device"))
    batch, micro = profile["batch"], profile["micro_batch"]
    if batch < 1 or micro < 1 or batch % micro:
        raise SystemExit(f"batch={batch} 必须是 micro_batch={micro} 的正整数倍。")
    if world > 1 and micro % world:
        raise SystemExit(
            f"DDP 要求 micro-batch 能被 GPU 数整除: micro={micro}, GPUs={world}。"
        )


def print_plan(models, profile, args):
    print(f"data={DATA_CONFIG}")
    print(f"profile={PROFILE_CONFIG}")
    print("settings=" + ", ".join(f"{k}={v}" for k, v in profile.items()))
    settings = training_kwargs(profile)
    world = device_count(profile.get("device"))
    print(
        f"training: effective_batch={profile['batch']}, micro_batch={settings['batch']}, "
        f"accumulation_steps={settings['gradient_accumulation']}, "
        f"warmup_micro_steps={settings['warmup_epochs']}, GPUs={world}"
    )
    for name, (yaml_name, project_dir) in models.items():
        cfg = RTDETR_MODEL_DIR / yaml_name
        out = args.project / project_dir / "train" / f"rt-detr-{yaml_name[:-5]}"
        mode = "resume" if args.resume else ("weights" if args.weights else "from-scratch")
        print(f"{name}: config={cfg}, project_dir={args.project / project_dir}, out={out}, init={mode}")


def train(models, profile, args):
    from ultralytics import RTDETR

    if args.weights is not None and len(models) != 1:
        raise SystemExit("--weights 仅能与单个模型配合使用。")
    if args.resume and len(models) != 1:
        raise SystemExit("--resume 仅能与单个模型配合使用。")

    settings = training_kwargs(profile)
    for name, (yaml_name, project_dir) in models.items():
        config = RTDETR_MODEL_DIR / yaml_name
        project = args.project / project_dir
        run_name = f"rt-detr-{yaml_name[:-5]}"
        out_dir = project / "train" / run_name
        if out_dir.exists() and not args.resume and not args.force:
            print(f"[skip] 已存在运行目录 {out_dir}; 用 --resume 或 --force 继续。")
            continue

        print(f"\n{'=' * 24} TRAIN {name} ({yaml_name}) {'=' * 24}")
        model = RTDETR(str(config))
        if args.weights is not None:
            model.load(str(args.weights))
        model.train(
            data=str(DATA_CONFIG),
            project=str(project),
            name=run_name,
            resume=args.resume,
            **settings,
        )


def main():
    args = parse_args()
    models = selected(args.model)
    profile = load_profile(args)
    validate_profile(profile)
    if not args.train:
        print_plan(models, profile, args)
        return
    train(models, profile, args)


if __name__ == "__main__":
    main()
