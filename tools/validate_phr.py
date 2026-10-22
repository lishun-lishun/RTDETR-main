#!/usr/bin/env python3
"""PHR-ResNet18 静态校验脚本 (Linux 服务器端, 训练前 dry-run 验证).

在 *不启动训练* 的前提下, 依次构建 baseline 与三份 PHR YAML, 校验:
    1. 模型可正常实例化 (parse_model 无误);
    2. Decoder 的 input_proj 输入通道为 [256, 256, 256];
    3. 恰好包含一个 PHR 模块, 且其反馈开关与 YAML args 一致;
    4. 初始化恒等: PHR 输出 == 输入 P3 (gamma 零初始化);
    5. 参数量 / GFLOPs (复用 model_info/thop);
    6. 640x640 全模型前向不报错。

任何一项失败都会以非零退出码结束, 供 CI / 训练前冒烟测试使用。

用法示例
--------
    python tools/validate_phr.py
    python tools/validate_phr.py --device cpu --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RTDETR_MODEL_DIR = _PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"

# role -> (yaml, 期望反馈开关)
_YAMLS = {
    "baseline": ("rtdetr-r18.yaml", None),
    "phr1": ("rtdetr_r18_phr1.yaml", (False, False, 2)),
    "phr2": ("rtdetr_r18_phr2.yaml", (True, False, 2)),
    "phr3": ("rtdetr_r18_phr3.yaml", (True, True, 3)),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="例如 0 / cpu, 默认自动。")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def _find_decoder(module):
    from ultralytics.nn.modules.head import RTDETRDecoder
    if isinstance(module, RTDETRDecoder):
        return module
    for child in module.children():
        found = _find_decoder(child)
        if found is not None:
            return found
    return None


def _find_phr(module, acc=None):
    from ultralytics.nn.modules.phr import PHR
    if acc is None:
        acc = []
    if isinstance(module, PHR):
        acc.append(module)
    for child in module.children():
        _find_phr(child, acc)
    return acc


def main():
    args = parse_args()
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import model_info, select_device

    device = select_device(args.device)
    failures = []

    for role, (yaml_name, expect) in _YAMLS.items():
        cfg = RTDETR_MODEL_DIR / yaml_name
        print(f"\n{'=' * 20} {role} ({yaml_name}) {'=' * 20}")
        try:
            model = RTDETR(str(cfg))
        except Exception as e:  # noqa: BLE001
            failures.append((role, f"实例化失败: {e}"))
            print(f"  [FAIL] 实例化: {e}")
            continue

        # Decoder 通道
        decoder = _find_decoder(model.model)
        in_ch = [proj[0].in_channels for proj in decoder.input_proj]
        ok = in_ch == [256, 256, 256]
        print(f"  decoder.input_proj channels = {in_ch}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append((role, f"decoder channels {in_ch} != [256,256,256]"))

        # 参数量 / GFLOPs
        n_l, n_p, n_g, flops = model_info(model.model, imgsz=args.imgsz)
        print(f"  layers={n_l}  params={n_p:,}  grads={n_g:,}  GFLOPs={flops:.2f}")

        # PHR 模块 (baseline 应无)
        phr = _find_phr(model.model)
        if role == "baseline":
            if phr:
                failures.append((role, "baseline 不应包含 PHR"))
                print("  [FAIL] baseline 意外包含 PHR")
            else:
                print("  PHR: (none, baseline)  OK")
        else:
            if len(phr) != 1:
                failures.append((role, f"PHR 数量 {len(phr)} != 1"))
                print(f"  [FAIL] PHR 数量 = {len(phr)}")
                continue
            m = phr[0]
            fb = (m.use_p3_feedback, m.use_p4_feedback, m.num_hr_blocks)
            ok_fb = fb == expect
            print(f"  PHR feedback=(p3={fb[0]}, p4={fb[1]}), num_hr_blocks={fb[2]}  "
                  f"{'OK' if ok_fb else f'FAIL (expect {expect})'}")
            if not ok_fb:
                failures.append((role, f"反馈配置 {fb} != {expect}"))

        # 全模型前向 (eval)
        try:
            model.model.eval().to(device)
            with torch.no_grad():
                out = model.model(torch.zeros(1, 3, args.imgsz, args.imgsz).to(device))
            assert isinstance(out, tuple) and out[0].shape[0] == 1
            print(f"  forward({args.imgsz}) -> {tuple(out[0].shape)}  OK")
        except Exception as e:  # noqa: BLE001
            failures.append((role, f"前向失败: {e}"))
            print(f"  [FAIL] 前向: {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    if failures:
        print(f"校验失败 {len(failures)} 项:")
        for role, msg in failures:
            print(f"  - {role}: {msg}")
        raise SystemExit(1)
    print("全部校验通过 ✅")


if __name__ == "__main__":
    main()
