"""Fuse SA-Rep branches in a trained checkpoint and verify the saved model."""
import argparse
import copy
import json
from pathlib import Path

from sarep_spasff_common import local_package


def tensors(value):
    import torch
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def errors(reference, actual):
    pairs = list(zip(tensors(reference), tensors(actual)))
    if not pairs or len(tensors(reference)) != len(tensors(actual)):
        raise RuntimeError('checkpoint outputs have different structures')
    max_abs = max((a.float() - b.float()).abs().max().item() for a, b in pairs)
    mean_abs = sum((a.float() - b.float()).abs().mean().item() for a, b in pairs) / len(pairs)
    relative = max(((a.float() - b.float()).norm() /
                    a.float().norm().clamp_min(1e-8)).item() for a, b in pairs)
    return {'max_abs_error': max_abs, 'mean_abs_error': mean_abs,
            'max_relative_error': relative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=128, help='Multiple-of-32 equivalence input')
    args = parser.parse_args()
    if not args.weights.is_file() or args.output.exists() or args.size % 32:
        parser.error('source must exist, output must be new, size must be a multiple of 32')
    report_path = args.output.with_suffix('.json')
    if report_path.exists():
        parser.error(f'deployment report already exists: {report_path}')
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules.sarep_spasff import SARepConv, switch_sarep_to_deploy
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    source = checkpoint.get('ema') or checkpoint.get('model')
    if source is None:
        raise RuntimeError('checkpoint has no EMA or model')
    source = copy.deepcopy(source).float().eval()
    if not any(isinstance(m, SARepConv) for m in source.modules()):
        raise RuntimeError('no SARepConv in checkpoint')
    torch.manual_seed(123)
    input_tensor = torch.randn(1, 3, args.size, args.size)
    with torch.inference_mode():
        before = source(input_tensor)
    deploy = copy.deepcopy(source)
    converted = switch_sarep_to_deploy(deploy)
    with torch.inference_mode():
        after = deploy(input_tensor)
    report = {'converted_blocks': converted, 'in_memory': errors(before, after)}
    if report['in_memory']['max_abs_error'] > 1e-3 and report['in_memory']['max_relative_error'] > 1e-4:
        raise RuntimeError(f'deploy conversion differs: {report}')
    saved = dict(checkpoint)
    saved['ema'] = deploy.half() if checkpoint.get('ema') is not None else None
    saved['model'] = deploy.half() if checkpoint.get('model') is not None else None
    saved['optimizer'] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + '.tmp.pt')
    if temporary.exists():
        parser.error(f'temporary output already exists: {temporary}')
    torch.save(saved, temporary)
    try:
        loaded = RTDETR(str(temporary)).model.float().eval()
        with torch.inference_mode():
            reloaded = loaded(input_tensor)
        report['reloaded'] = errors(before, reloaded)
        if (report['reloaded']['max_abs_error'] > 1e-3 and
                report['reloaded']['max_relative_error'] > 1e-4):
            raise RuntimeError(f'saved deploy checkpoint differs: {report}')
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    report['source'] = str(args.weights)
    report['deploy'] = str(args.output)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
