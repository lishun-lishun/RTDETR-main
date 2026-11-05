"""Collect DLA34 level RMS, numerical and gradient diagnostics."""

import argparse
import json
from pathlib import Path

from dla34_common import CONFIGS, config_path, local_package, write_json


def stats(value):
    value = value.detach().float()
    return {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item(),
            "rms": value.square().mean().sqrt().item(), "nan_count": int(value.isnan().sum()),
            "inf_count": int(value.isinf().sum())}


def tensors(value):
    import torch
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(tensors(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensors(item))
        return result
    return []


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
    from ultralytics.nn.modules import DLA34Backbone
    from ultralytics.nn.tasks import RTDETRDetectionModel

    network = (RTDETR(str(args.weights)).model if args.weights else
               RTDETRDetectionModel(str(config_path(args.model)), nc=1, verbose=False))
    network = network.to(args.device).float().eval()
    backbone = next(module for module in network.modules() if isinstance(module, DLA34Backbone))
    captured = {}
    handle = backbone.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__(
            "backbone_outputs", [stats(value) for value in output]))
    sample = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
    with torch.inference_mode():
        network(sample)
    handle.remove()
    report = {"source": str(args.weights or config_path(args.model)), "input_size": args.input_size,
              "backbone_outputs": captured.get("backbone_outputs")}
    if not args.no_gradient:
        network.zero_grad(set_to_none=True)
        network.train()
        output = network(sample)
        values = tensors(output)
        if not values:
            raise RuntimeError("model forward returned no tensor outputs for gradient diagnostic")
        loss = sum(item.float().square().mean() for item in values)
        loss.backward()
        report["training_diagnostics"] = {
            "loss": float(loss.detach()),
            "level_gradient_norms": {
                name: next(parameter.grad.norm().item() for parameter in getattr(backbone, name).parameters()
                           if parameter.grad is not None)
                for name in ("level3", "level4", "level5")
            },
            "tree_parameter_count": sum(parameter.numel() for parameter in backbone.parameters()),
            "nan_count": sum(int(item.isnan().sum()) for item in values),
            "inf_count": sum(int(item.isinf().sum()) for item in values),
        }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
