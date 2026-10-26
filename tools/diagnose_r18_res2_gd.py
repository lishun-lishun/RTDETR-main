"""Record Res2 group and GD injection behavior from a checkpoint and image."""
import argparse
import json
from pathlib import Path

from res2_gd_common import local_package


def stats(tensor):
    value = tensor.detach().float()
    return {'mean': value.mean().item(), 'std': value.std().item(),
            'rms': value.square().mean().sqrt().item(),
            'shape': list(value.shape)}


def residual_ratio(local, correction):
    local = local.detach().float()
    correction = correction.detach().float()
    return (correction.square().mean().sqrt() /
            local.square().mean().sqrt().clamp_min(1e-8)).item()


def image_tensor(path, device):
    import numpy as np
    from PIL import Image
    import torch

    with Image.open(path) as image:
        original_width, original_height = image.size
        scale = min(640 / original_width, 640 / original_height)
        width, height = round(original_width * scale), round(original_height * scale)
        left, top = (640 - width) // 2, (640 - height) // 2
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(image.convert('RGB').resize((width, height)), (left, top))
        array = np.asarray(canvas).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    return tensor, (scale, left, top)


def region_rms(features, annotations, image_record, geometry):
    import math
    import torch

    from evaluate_r18_sarep_spasff import group_name

    scale, left, top = geometry
    output = {}
    for level, feature in zip(('P3', 'P4', 'P5'), features):
        feature = feature.detach().float()[0]
        masks = {name: torch.zeros(feature.shape[-2:], dtype=torch.bool,
                                   device=feature.device)
                 for name in ('Tiny', 'Small', 'Medium', 'Large')}
        for annotation in annotations:
            if annotation.get('iscrowd', 0):
                continue
            x, y, width, height = annotation['bbox']
            x1 = max(0, int((x * scale + left) / 640 * feature.shape[-1]))
            y1 = max(0, int((y * scale + top) / 640 * feature.shape[-2]))
            x2 = min(feature.shape[-1], max(x1 + 1, math.ceil(
                ((x + width) * scale + left) / 640 * feature.shape[-1])))
            y2 = min(feature.shape[-2], max(y1 + 1, math.ceil(
                ((y + height) * scale + top) / 640 * feature.shape[-2])))
            masks[group_name(annotation, image_record)][y1:y2, x1:x2] = True
        output[level] = {name: feature[:, mask].square().mean().sqrt().item()
                         if mask.any() else None for name, mask in masks.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-gradient', action='store_true',
                        help='Skip the 128px diagnostic backward pass')
    args = parser.parse_args()
    if not args.weights.is_file() or not args.image.is_file() or args.output_dir.exists():
        parser.error('weights/image must exist and output directory must be new')
    if args.annotations and not args.annotations.is_file():
        parser.error('annotations file is missing')
    local_package()
    import torch
    from torch.nn import functional as F
    from ultralytics import RTDETR
    from ultralytics.nn.modules import GDInject, LiteGDNeck, Res2Conv

    network = RTDETR(str(args.weights)).model.to(args.device).eval()
    input_tensor, geometry = image_tensor(args.image, args.device)
    captured = {'injections': {}, 'residual_tensors': {}}
    handles = []

    def capture_neck(_module, _inputs, output):
        captured['neck_features'] = [x.detach() for x in output]
        captured['neck_local'] = [refine(project(x)).detach()
                                  for x, project, refine in
                                  zip(_inputs[0], _module.base, _module.local_refine)]

    def capture_stage(_module, _inputs, output):
        captured['stage3'] = output.detach()

    stage_hook_set = False
    for name, module in network.named_modules():
        if isinstance(module, LiteGDNeck):
            handles.append(module.register_forward_hook(capture_neck))
        if isinstance(module, Res2Conv) and not stage_hook_set:
            handles.append(module.register_forward_hook(capture_stage))
            stage_hook_set = True
        if isinstance(module, GDInject):
            def capture_inject(_module, inputs, output, key=name):
                captured['residual_tensors'][key] = (output - inputs[0]).detach()
                captured['injections'][key] = {
                    'effective_gain': (_module.gain_max *
                                       torch.tanh(_module.raw_gain)).item(),
                    'residual_rms_ratio': residual_ratio(inputs[0], output - inputs[0]),
                    'local_rms': stats(inputs[0])['rms'],
                    'residual_rms': stats(output - inputs[0])['rms']}
            handles.append(module.register_forward_hook(capture_inject))
    with torch.inference_mode():
        network(input_tensor)
    for handle in handles:
        handle.remove()

    res2_report = {'weights': str(args.weights), 'image': str(args.image),
                   'groups': [], 'stage3_feature':
                   stats(captured['stage3']) if 'stage3' in captured else None,
                   'gradient_norm': None}
    for name, module in network.named_modules():
        if not isinstance(module, Res2Conv):
            continue
        outputs = module.last_group_outputs or []
        input_groups = module.last_hierarchical_inputs or []
        correlations = [F.cosine_similarity(
            outputs[index].float().flatten(1),
            outputs[index - 1].float().flatten(1), dim=1).mean().item()
            for index in range(1, len(outputs))]
        connections = [residual_ratio(
            input_groups[index - 1] - outputs[index - 1], outputs[index - 1])
            for index in range(2, len(outputs))] if module.hierarchical else []
        res2_report['groups'].append({
            'module': name, 'scale': module.scale, 'width': module.width,
            'hierarchical': module.hierarchical,
            'feature_statistics': [stats(group) for group in outputs],
            'adjacent_group_cosine': correlations,
            'hierarchical_previous_output_to_current_chunk_rms_ratio': connections})

    gd_report = {'weights': str(args.weights), 'image': str(args.image),
                 'high_gather': None, 'low_gather': None,
                 'injections': captured['injections'],
                 'per_scale_residual_rms_ratio': None,
                 'output_feature_statistics': [],
                 'size_group_residual_rms': None}
    necks = [module for module in network.modules() if isinstance(module, LiteGDNeck)]
    if necks:
        neck = necks[0]
        gd_report['mode'] = neck.mode
        gd_report['high_gather'] = stats(neck.last_high) if neck.last_high is not None else None
        gd_report['low_gather'] = stats(neck.last_low) if neck.last_low is not None else None
        gd_report['output_feature_statistics'] = [stats(x) for x in captured['neck_features']]
        gd_report['per_scale_residual_rms_ratio'] = {
            level: residual_ratio(local, output - local)
            for level, local, output in zip(
                ('P3', 'P4', 'P5'), captured['neck_local'],
                captured['neck_features'])}
        if args.annotations:
            from pycocotools.coco import COCO
            gt = COCO(str(args.annotations))
            matched = [image for image in gt.dataset['images']
                       if Path(image['file_name']).stem == args.image.stem]
            if len(matched) != 1:
                raise RuntimeError('image stem must match exactly one COCO image')
            suffixes = (('low_inject.0',),
                        ('high_inject4', 'low_inject.1'),
                        ('high_inject5', 'low_inject.2'))
            residuals = []
            for feature, level_suffixes in zip(captured['neck_features'], suffixes):
                contributions = [value for key, value in captured['residual_tensors'].items()
                                 if key.endswith(level_suffixes)]
                residuals.append(sum(contributions, torch.zeros_like(feature)))
            gd_report['size_group_residual_rms'] = region_rms(
                residuals, gt.imgToAnns[matched[0]['id']],
                matched[0], geometry)
    if not args.no_gradient:
        network.zero_grad(set_to_none=True)
        smaller = F.interpolate(input_tensor, size=(128, 128), mode='bilinear',
                                align_corners=False)
        result = network(smaller)
        prediction = result[0] if isinstance(result, (tuple, list)) else result
        prediction.float().square().mean().backward()
        res2_report['gradient_norm'] = {
            name: module.pre.conv.weight.grad.float().norm().item()
            if module.pre.conv.weight.grad is not None else None
            for name, module in network.named_modules() if isinstance(module, Res2Conv)}
        gd_report['gradient_norm'] = {
            name: module.base[0][0].weight.grad.float().norm().item()
            if module.base[0][0].weight.grad is not None else None
            for name, module in network.named_modules() if isinstance(module, LiteGDNeck)}
    args.output_dir.mkdir(parents=True)
    for filename, report in (('res2basic_statistics.json', res2_report),
                             ('litegd_statistics.json', gd_report)):
        (args.output_dir / filename).write_text(
            json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'res2': len(res2_report['groups']), 'gd': len(necks),
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
