"""Collect MIRNet branch RMS, exchange residuals and gradient diagnostics."""

import argparse
import json
from pathlib import Path

from mirnet_det_common import CONFIGS, config_path, local_package, write_json


def stats(value):
    value = value.detach().float()
    return {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item(),
            "rms": value.square().mean().sqrt().item(), "nan_count": int(value.isnan().sum()),
            "inf_count": int(value.isinf().sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--model", choices=CONFIGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-gradient", action="store_true")
    args = parser.parse_args()
    if args.weights and not args.weights.is_file():
        parser.error("weights file does not exist")
    if args.output.exists():
        parser.error("output already exists")
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules import DualResolutionExchange, MIRDetBackbone, TriResolutionExchange
    from ultralytics.nn.tasks import RTDETRDetectionModel

    network = (RTDETR(str(args.weights)).model if args.weights else
               RTDETRDetectionModel(str(config_path(args.model)), nc=1, verbose=False))
    network = network.to(args.device).float().eval()
    captured = {}
    handles = []

    def capture_stage(_module, _inputs, output):
        captured["backbone_outputs"] = [stats(value) for value in output]

    handles.append(next(module for module in network.modules() if isinstance(module, MIRDetBackbone)).register_forward_hook(capture_stage))
    for name, module in network.named_modules():
        if isinstance(module, (DualResolutionExchange, TriResolutionExchange)):
            handles.append(module.register_forward_hook(
                lambda _m, _i, _o, name=name: captured.setdefault("exchanges", {}).__setitem__(
                    name, {key: stats(value) for key, value in _m.last_residuals.items()})))
    sample = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
    with torch.inference_mode():
        network(sample)
    for handle in handles:
        handle.remove()
    report = {"source": str(args.weights or config_path(args.model)), "input_size": args.input_size,
              "backbone_outputs": captured.get("backbone_outputs"), "exchanges": captured.get("exchanges", {})}
    if not args.no_gradient:
        network.zero_grad(set_to_none=True)
        network.train()
        output = network(sample)
        tensors = output if isinstance(output, (tuple, list)) else (output,)
        loss = sum(item.float().square().mean() for item in tensors if torch.is_tensor(item))
        loss.backward()
        backbone = next(module for module in network.modules() if isinstance(module, MIRDetBackbone))
        report["training_diagnostics"] = {
            "loss": float(loss.detach()),
            "branch_gradient_norms": {
                "s8": backbone.s8_blocks[0].expand.conv.weight.grad.norm().item(),
                "s16": backbone.s16_generate.conv.conv.weight.grad.norm().item(),
                "s32": backbone.s32_generate.conv.conv.weight.grad.norm().item(),
            },
            "nan_count": sum(int(item.isnan().sum()) for item in tensors if torch.is_tensor(item)),
            "inf_count": sum(int(item.isinf().sum()) for item in tensors if torch.is_tensor(item)),
        }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
