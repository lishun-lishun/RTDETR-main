"""Record LDW stage RMS, gradient and feature-shape diagnostics."""

import argparse
import json
from pathlib import Path

from ldw_r18_common import CONFIGS, config_path, local_package, preflight, write_json


def tensor_stats(value):
    value = value.detach().float()
    return {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item(),
            "rms": value.square().mean().sqrt().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--model", choices=CONFIGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--no-gradient", action="store_true")
    args = parser.parse_args()
    if args.weights and not args.weights.is_file():
        parser.error("weights file does not exist")
    if args.output.exists():
        parser.error("output already exists")
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if args.weights:
        network = RTDETR(str(args.weights)).model
    else:
        network = RTDETRDetectionModel(str(config_path(args.model)), nc=1, verbose=False)
    network = network.to(args.device).float().eval()
    captured = {}
    handles = []
    for index in (5, 6, 7):
        handles.append(network.model[index].register_forward_hook(
            lambda _m, inputs, output, index=index: captured.__setitem__(index, {
                "input": tensor_stats(inputs[0]), "output": tensor_stats(output)})))
    sample = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
    with torch.inference_mode():
        network(sample)
    for handle in handles:
        handle.remove()
    report = {"source": str(args.weights or config_path(args.model)), "input_size": args.input_size,
              "stage3": captured.get(5), "stage4": captured.get(6), "stage5": captured.get(7),
              "feature_shapes": {str(k): v["output"]["shape"] for k, v in captured.items()},
              "preflight": preflight(network, args.input_size)}
    if not args.no_gradient:
        network.zero_grad(set_to_none=True)
        network.train()
        output = network(sample)
        # RT-DETR inference returns a tuple/list; a scalar reduction still
        # exercises every trainable backbone and decoder parameter.
        tensors = output if isinstance(output, (tuple, list)) else (output,)
        loss = sum(x.float().square().mean() for x in tensors if torch.is_tensor(x))
        loss.backward()
        def first_gradient(module):
            for parameter in module.parameters():
                if parameter.grad is not None:
                    return parameter.grad.detach().float().norm().item()
            return None

        report["gradient_norms"] = {
            "stage3": first_gradient(network.model[5]),
            "stage4": first_gradient(network.model[6]),
            "stage5": first_gradient(network.model[7]),
            "neck_decoder": first_gradient(network.model[-1]),
        }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
