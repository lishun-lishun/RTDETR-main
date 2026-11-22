import os, sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(repo_root / '.yolo')
os.environ['MPLCONFIGDIR'] = str(repo_root / '.yolo')
sys.path.insert(0, str(repo_root / '.pypkg'))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, r'E:\envs\my_env\Lib\site-packages')
sys.path.append(r'E:\envs\isaaclab\Lib\site-packages')
sys.path.append(r'E:\envs\RareCLIP\Lib\site-packages')

import torch
import torch.nn as nn
from ultralytics import RTDETR

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running FP64 equivalence check on device: {device}')

cfgs = [
    ('Exp1_BaseHRNet', 'RTDETR_HRNet_Lite_Suite/configs/exp1_base_hrnet_w18.yaml'),
    ('Exp6_StarHRNet', 'RTDETR_HRNet_Lite_Suite/configs/exp6_hrnet_starrep.yaml'),
    ('Exp10_StarHR_Speed', 'RTDETR_HRNet_Lite_Suite/configs/exp10_starhr_speed.yaml'),
    ('Exp11_StarHR_192Speed', 'RTDETR_HRNet_Lite_Suite/configs/exp11_starhr_192speed.yaml'),
    ('Exp12_StarHR_Balanced', 'RTDETR_HRNet_Lite_Suite/configs/exp12_starhr_balanced.yaml'),
]

results = []
for name, cfg_path in cfgs:
    print(f'\n========== Testing FP64 equivalence for {name} ==========')
    det = RTDETR(str(repo_root / cfg_path))
    model = det.model

    # Set non-trivial random BN stats to strictly stress-test folding
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-1.0, 1.0)
            m.running_var.uniform_(0.5, 2.0)
            if m.weight is not None:
                m.weight.data.uniform_(0.5, 1.5)
            if m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)

    model = model.to(device).double().eval()
    torch.manual_seed(42)
    x = torch.randn(1, 3, 320, 320, device=device, dtype=torch.float64)

    with torch.no_grad():
        out_before = model(x)

    model.fuse()
    grads = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        out_after = model(x)

    if isinstance(out_before, (tuple, list)):
        diffs = [(a - b).abs().max().item() for a, b in zip(out_before, out_after) if isinstance(a, torch.Tensor)]
        max_diff = max(diffs)
    elif isinstance(out_before, torch.Tensor):
        max_diff = (out_before - out_after).abs().max().item()
    else:
        out_b = out_before[0] if isinstance(out_before, tuple) else out_before
        out_a = out_after[0] if isinstance(out_after, tuple) else out_after
        max_diff = (out_b - out_a).abs().max().item()

    tol = 1e-9 if 'StarHR' in name else 1e-10
    print(f'[{name}] max_abs_diff = {max_diff:.3e} (tol={tol:.0e}), remaining grads = {grads}')
    assert max_diff < tol, f'{name} failed tolerance: {max_diff} >= {tol}'
    assert grads == 0, f'{name} has unclosed grads: {grads}'
    results.append((name, max_diff, grads))

print('\n' + '=' * 60)
print(f"{'Experiment':<25} | {'Max FP64 Diff':>15} | {'Grads':>6}")
print('-' * 60)
for name, md, gr in results:
    print(f"{name:<25} | {md:>15.3e} | {gr:>6d}")
print('=' * 60)
print('ALL 5 MODELS 100% PASSED FP64 MATHEMATICAL EQUIVALENCE CHECK!')
