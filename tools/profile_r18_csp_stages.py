"""Profile isolated original/CSP stages with the same PyTorch FLOPs counter."""
import argparse
import ast
from collections import OrderedDict
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / 'ultralytics/nn/modules'


def load_modules():
    tree = ast.parse((MODULE_DIR / 'block.py').read_text(encoding='utf-8'))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in
                {'get_activation', 'ConvNormLayer', 'BasicBlock'}]
    namespace = {'nn': nn, 'torch': torch, 'OrderedDict': OrderedDict}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(MODULE_DIR / 'block.py'), 'exec'), namespace)
    package = types.ModuleType('_csp_stage_profile')
    package.__path__ = [str(MODULE_DIR)]
    sys.modules[package.__name__] = package
    block = types.ModuleType(package.__name__ + '.block')
    for key in ('ConvNormLayer', 'BasicBlock'):
        setattr(block, key, namespace[key])
    sys.modules[block.__name__] = block
    return namespace['BasicBlock'], importlib.import_module(package.__name__ + '.csp_context')


def measure(model, sample):
    model.eval()
    with torch.inference_mode(), torch.profiler.profile(with_flops=True) as profile:
        output = model(sample)
    flops = sum(event.flops for event in profile.key_averages())
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
            'supported_flops_lower_bound': flops,
            'supported_gflops_lower_bound': flops / 1e9,
            'supported_macs_lower_bound': flops / 2,
            'output_shape': list(output.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    torch.set_num_threads(2)
    BasicBlock, csp = load_modules()
    report = {'protocol': 'PyTorch profiler supported ops; isolated stages, batch=1, FP32',
              'input_size': 640, 'stages': {}}
    for stage_num, c1, c2, size in ((3, 64, 128, 160), (4, 128, 256, 80)):
        original = nn.Sequential(BasicBlock(c1, c2, 2, False),
                                 BasicBlock(c2, c2, 1, True))
        modified = (csp.CSPResidualStage(c1, c2) if stage_num == 3 else
                    csp.CSPContextStage(c1, c2))
        sample = torch.randn(1, c1, size, size)
        old = measure(original, sample)
        new = measure(modified, sample)
        report['stages'][str(stage_num)] = {
            'original': old, 'csp': new,
            'parameter_delta': new['parameters'] - old['parameters'],
            'supported_flops_delta': new['supported_flops_lower_bound'] -
            old['supported_flops_lower_bound']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
