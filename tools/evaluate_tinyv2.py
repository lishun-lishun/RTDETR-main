"""Evaluate one TinyV2 checkpoint on the validation split by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r18_targeted_common import DATA, ROOT, local_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error('weights and data YAML must exist')
    output = ROOT / 'runs' / 'tinyv2_eval' / args.name
    if output.exists():
        parser.error(f'evaluation output already exists: {output}')
    local_package()
    from ultralytics import RTDETR

    result = RTDETR(str(args.weights)).val(
        data=str(args.data), split=args.split, imgsz=640, batch=args.batch,
        device=args.device, max_det=300, save_json=True,
        project=str(output.parent), name=output.name, exist_ok=False)
    metric = result.box
    payload = {
        'weights': str(args.weights.resolve()), 'split': args.split,
        'data': str(args.data.resolve()), 'device': args.device,
        'AP50': float(metric.map50), 'AP75': float(metric.map75),
        'AP50:95': float(metric.map),
        'speed_ms': getattr(result, 'speed', {}),
        'note': 'Small/Medium/Large AP requires COCO area-stratified evaluation; not reported by this evaluator.',
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(json.dumps(payload, indent=2, default=str))


if __name__ == '__main__':
    main()
