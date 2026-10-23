"""Inspect LAF sampling offsets and isolated P3 residual on one image."""
import argparse
import json
from pathlib import Path

import torch
from train_rb_hrnet_laf import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.image.is_file():
        parser.error('weights and image must exist')
    if args.output.exists():
        parser.error('output already exists')
    import numpy as np
    from PIL import Image
    from ultralytics import RTDETR
    from ultralytics.nn.modules.laf import DynamicUpsample, IsolatedLAF

    network = RTDETR(str(args.weights)).model.to(args.device).eval()
    blocks = [m for m in network.modules() if isinstance(m, IsolatedLAF)]
    samplers = [m for m in network.modules() if isinstance(m, DynamicUpsample)]
    if len(samplers) != 1:
        raise RuntimeError(f'Expected exactly one LAF sampler, found {len(samplers)}')
    before_after = {}
    def hook(_module, inputs, output):
        f3 = inputs[0][0] if isinstance(inputs[0], (list, tuple)) else inputs[0]
        before_after['difference_rms'] = ((output - f3).float().square().mean()
                                          .sqrt().item())
        before_after['relative_residual_norm'] = (
            (output - f3).float().norm() / f3.float().norm().clamp_min(1e-8)).item()
    handle = blocks[0].register_forward_hook(hook) if blocks else None
    image = Image.open(args.image).convert('RGB').resize((640, 640))
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(args.device).float() / 255
    with torch.inference_mode():
        network(tensor)
    if handle:
        handle.remove()
    sampler = samplers[0]
    offset = sampler.last_offset.float()
    result = {'offset_mean': offset.mean().item(),
              'offset_std': offset.std().item(),
              'offset_max_abs': offset.abs().max().item(),
              'offset_saturation_ratio': (
                  offset.abs() >= 0.95 * sampler.max_offset).float().mean().item(),
              'image': str(args.image), **before_after}
    if blocks:
        result['effective_alpha'] = (
            blocks[0].alpha_max * torch.tanh(blocks[0].alpha_raw)).item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
