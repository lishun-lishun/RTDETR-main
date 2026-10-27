"""Record CSP branch magnitudes, gains, gradients, and optional object regions."""
import argparse
import json
from pathlib import Path

from csp_context_common import local_package


def statistics(value):
    tensor = value.detach().float()
    return {'shape': list(tensor.shape), 'mean': tensor.mean().item(),
            'std': tensor.std().item(), 'rms': tensor.square().mean().sqrt().item()}


def load_image(path, device):
    import numpy as np
    from PIL import Image
    import torch

    with Image.open(path) as image:
        width, height = image.size
        scale = min(640 / width, 640 / height)
        new_width, new_height = round(width * scale), round(height * scale)
        left, top = (640 - new_width) // 2, (640 - new_height) // 2
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(image.convert('RGB').resize((new_width, new_height)),
                     (left, top))
        array = np.asarray(canvas).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    return tensor, (scale, left, top)


def region_statistics(feature, annotations, image, geometry):
    import math
    import torch

    from evaluate_r18_sarep_spasff import group_name

    scale, left, top = geometry
    feature = feature.detach().float()[0]
    masks = {key: torch.zeros(feature.shape[-2:], dtype=torch.bool,
                              device=feature.device)
             for key in ('Tiny', 'Small', 'Medium', 'Large')}
    for ann in annotations:
        if ann.get('iscrowd', 0):
            continue
        x, y, width, height = ann['bbox']
        x1 = max(0, int((x * scale + left) / 640 * feature.shape[-1]))
        y1 = max(0, int((y * scale + top) / 640 * feature.shape[-2]))
        x2 = min(feature.shape[-1], max(x1 + 1, math.ceil(
            ((x + width) * scale + left) / 640 * feature.shape[-1])))
        y2 = min(feature.shape[-2], max(y1 + 1, math.ceil(
            ((y + height) * scale + top) / 640 * feature.shape[-2])))
        masks[group_name(ann, image)][y1:y2, x1:x2] = True
    foreground = torch.stack(tuple(masks.values())).any(dim=0)
    rms = lambda mask: feature[:, mask].square().mean().sqrt().item() if mask.any() else None
    return {'size_group_rms': {key: rms(mask) for key, mask in masks.items()},
            'foreground_rms': rms(foreground),
            'background_rms': rms(~foreground)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-gradient', action='store_true')
    args = parser.parse_args()
    if not args.weights.is_file() or not args.image.is_file() or args.output.exists():
        parser.error('weights/image must exist and output must be new')
    if args.annotations and not args.annotations.is_file():
        parser.error('annotations file is missing')
    local_package()
    import torch
    from torch.nn import functional as F
    from ultralytics import RTDETR
    from ultralytics.nn.modules import CSPContextStage, CSPResidualStage

    network = RTDETR(str(args.weights)).model.to(args.device).eval()
    sample, geometry = load_image(args.image, args.device)
    with torch.inference_mode():
        network(sample)
    stages = [(name, module) for name, module in network.named_modules()
              if isinstance(module, (CSPResidualStage, CSPContextStage))]
    report = {'weights': str(args.weights), 'image': str(args.image), 'stages': []}
    annotations, image_record = None, None
    if args.annotations:
        from pycocotools.coco import COCO
        gt = COCO(str(args.annotations))
        found = [image for image in gt.dataset['images']
                 if Path(image['file_name']).stem == args.image.stem]
        if len(found) != 1:
            raise RuntimeError('image stem must match one COCO image')
        image_record = found[0]
        annotations = gt.imgToAnns[image_record['id']]
    for name, stage in stages:
        entry = {'module': name, 'type': type(stage).__name__,
                 'processed_channels': stage.process_channels,
                 'bypass_channels': stage.bypass_channels,
                 'effective_gamma': stage.gamma.detach().item(),
                 'features': {key: statistics(value)
                              for key, value in stage.last_features.items()},
                 'gradient_norm': None}
        if annotations is not None:
            entry['object_regions'] = {
                key: region_statistics(stage.last_features[key], annotations,
                                       image_record, geometry)
                for key in ('processed', 'bypass', 'output')}
        report['stages'].append(entry)
    if not args.no_gradient:
        network.zero_grad(set_to_none=True)
        smaller = F.interpolate(sample, size=(128, 128), mode='bilinear',
                                align_corners=False)
        prediction = network(smaller)
        prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        prediction.float().square().mean().backward()
        for entry, (_, stage) in zip(report['stages'], stages):
            entry['gradient_norm'] = {
                'gamma': stage.raw_gamma.grad.float().norm().item()
                if stage.raw_gamma.grad is not None else None,
                'split_projection': stage.split_projection.conv.weight.grad.float().norm().item()
                if stage.split_projection.conv.weight.grad is not None else None,
                'fusion': stage.fusion.conv.weight.grad.float().norm().item()
                if stage.fusion.conv.weight.grad is not None else None,
                'processed_branch': sum(p.grad.float().norm().item()
                                        for p in stage.process.parameters()
                                        if p.grad is not None)}
            if isinstance(stage, CSPContextStage):
                entry['gradient_norm']['context_blocks_dwconv'] = [
                    block.dwconv.weight.grad.float().norm().item()
                    if block.dwconv.weight.grad is not None else None
                    for block in stage.process]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'stages': len(stages), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
