"""Record SA-Rep branch statistics and SP-ASFF weights for one image."""
import argparse
import copy
import json
from pathlib import Path

from sarep_spasff_common import local_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, help='Optional COCO GT JSON for foreground analysis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.image.is_file() or args.output.exists():
        parser.error('weights/image must exist and output must be new')
    local_package()
    import numpy as np
    from PIL import Image
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules.sarep_spasff import SARepConv, SPASFF, switch_sarep_to_deploy
    from export_sarep_deploy import errors
    network = RTDETR(str(args.weights)).model.to(args.device).eval()
    before_after = {}
    handles = []
    spasff = [module for module in network.modules() if isinstance(module, SPASFF)]
    if spasff:
        def capture(_module, inputs, output):
            f3 = inputs[0][0]
            residual = output - f3
            before_after.update({
                'difference_rms': residual.float().square().mean().sqrt().item(),
                'relative_residual_rms': (residual.float().square().mean().sqrt() /
                                          f3.float().square().mean().sqrt().clamp_min(1e-8)).item(),
                'relative_residual_l2': (residual.float().norm() /
                                         f3.float().norm().clamp_min(1e-8)).item()})
        handles.append(spasff[0].register_forward_hook(capture))
    with Image.open(args.image) as image:
        original_width, original_height = image.size
        scale = min(640 / original_width, 640 / original_height)
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)
        pad_x = (640 - resized_width) // 2
        pad_y = (640 - resized_height) // 2
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(image.convert('RGB').resize((resized_width, resized_height)),
                     (pad_x, pad_y))
        array = np.asarray(canvas).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(args.device).float() / 255
    with torch.inference_mode():
        original_output = network(tensor)
    for handle in handles:
        handle.remove()
    report = {'weights': str(args.weights), 'image': str(args.image),
              'sarep': [], 'spasff': before_after}
    for module in network.modules():
        if isinstance(module, SARepConv):
            kernel, bias = module.equivalent_kernel_bias()
            row = {'fused_kernel_norm': kernel.float().norm().item(),
                   'fused_kernel_std': kernel.float().std().item(),
                   'fused_bias_norm': bias.float().norm().item(),
                   'deployed': hasattr(module, 'reparam')}
            if not hasattr(module, 'reparam'):
                row['branch_kernel_norms'] = {
                    name: getattr(module, name)[0].weight.float().norm().item()
                    for name in ('branch_3x3', 'branch_1x3', 'branch_3x1')}
            report['sarep'].append(row)
    if report['sarep'] and not all(row['deployed'] for row in report['sarep']):
        converted = copy.deepcopy(network)
        switch_sarep_to_deploy(converted)
        with torch.inference_mode():
            report['sarep_train_deploy_output_error'] = errors(original_output, converted(tensor))
    if spasff:
        weights = spasff[0].last_weights.float().cpu()[0]
        report['spasff']['scale_weight_mean'] = weights.mean(dim=(1, 2)).tolist()
        report['spasff']['scale_weight_std'] = weights.std(dim=(1, 2)).tolist()
        report['spasff']['effective_gamma'] = (
            spasff[0].gamma_max * torch.tanh(spasff[0].raw_gamma)).item()
        if args.annotations:
            from pycocotools.coco import COCO
            from evaluate_r18_sarep_spasff import group_name
            gt = COCO(str(args.annotations))
            matched = [im for im in gt.dataset['images']
                       if Path(im['file_name']).stem == args.image.stem]
            if len(matched) != 1:
                raise RuntimeError('image stem must uniquely match one COCO annotation image')
            image = matched[0]
            masks = {name: torch.zeros_like(weights[0], dtype=torch.bool)
                     for name in ('Tiny', 'Small', 'Medium', 'Large')}
            for ann in gt.imgToAnns[image['id']]:
                if ann.get('iscrowd', 0):
                    continue
                x, y, w, h = ann['bbox']
                x1 = max(0, int((x * scale + pad_x) / 640 * weights.shape[-1]))
                y1 = max(0, int((y * scale + pad_y) / 640 * weights.shape[-2]))
                x2 = min(weights.shape[-1], max(x1 + 1, int(np.ceil(((x + w) * scale + pad_x) / 640 * weights.shape[-1]))))
                y2 = min(weights.shape[-2], max(y1 + 1, int(np.ceil(((y + h) * scale + pad_y) / 640 * weights.shape[-2]))))
                masks[group_name(ann, image)][y1:y2, x1:x2] = True
            foreground = torch.stack(tuple(masks.values())).any(dim=0)
            report['spasff']['background_weight_mean'] = weights[:, ~foreground].mean(dim=1).tolist() \
                if (~foreground).any() else None
            report['spasff']['foreground_weight_mean'] = weights[:, foreground].mean(dim=1).tolist() \
                if foreground.any() else None
            report['spasff']['size_group_weight_mean'] = {
                name: weights[:, mask].mean(dim=1).tolist() if mask.any() else None
                for name, mask in masks.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
