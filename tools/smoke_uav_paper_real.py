"""Real RTDETRDetectionModel smoke test; run on the configured GPU server."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import ultralytics
from ultralytics.nn.tasks import RTDETRDetectionModel

if Path(ultralytics.__file__).resolve() != ROOT / 'ultralytics' / '__init__.py':
    raise RuntimeError(f'Expected local Ultralytics in {ROOT}, got {ultralytics.__file__}')


CONFIGS = [
    'rtdetr-r18-inno1-wcasfp.yaml',
    'rtdetr-r18-inno2-faga.yaml',
    'rtdetr-r18-inno3-uadcr.yaml',
    'rtdetr-r18-inno4-sahnwd.yaml',
    'rtdetr-r18-combo-a.yaml',
    'rtdetr-r18-combo-b.yaml',
    'rtdetr-r18-uav-paper-full.yaml',
]


def batch_for(device, count, size):
    return {
        'img': torch.rand(2, 3, size, size, device=device),
        'cls': torch.zeros(count, 1, dtype=torch.long, device=device),
        'bboxes': torch.tensor([[.5, .5, .02, .02]] * count,
                              device=device).reshape(count, 4),
        'batch_idx': torch.tensor([i * 2 // count for i in range(count)],
                                  dtype=torch.long, device=device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    base = Path('ultralytics/cfg/models/rt-detr')
    for name in CONFIGS:
        model = RTDETRDetectionModel(str(base / name), nc=1, verbose=False).to(device)
        model.eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, args.size, args.size, device=device))
        assert result[0].shape == (1, 300, 5), (name, result[0].shape)
        model.train()
        for count, amp in ((0, False), (1, False), (3, True)):
            model.zero_grad(set_to_none=True)
            batch = batch_for(device, count, args.size)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
                loss, parts = model(batch)
            assert torch.isfinite(loss) and torch.isfinite(parts).all(), name
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            torch.save({'model': model, 'ema': deepcopy(model)}, path)
            restored = torch.load(path, map_location=device, weights_only=False)
            assert restored['model'].model[-1].num_queries == 300
        print(name, 'PASS', flush=True)


if __name__ == '__main__':
    main()
