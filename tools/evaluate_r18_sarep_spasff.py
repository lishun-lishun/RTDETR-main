"""Evaluate one checkpoint with COCO AP and input-640 size groups."""
import argparse
import copy
import csv
import json
from pathlib import Path

from sarep_spasff_common import CONFIGS, DATA, local_package


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def group_name(annotation, image):
    width, height = annotation['bbox'][2:4]
    ratio = min(640 / image['width'], 640 / image['height'])
    side = max(width, height) * ratio
    return 'Tiny' if side <= 8 else 'Small' if side <= 16 else 'Medium' if side <= 32 else 'Large'


def measure_groups(gt, predictions):
    from pycocotools.coco import COCO
    from evaluate_rb_hrnet_laf import coco_metrics
    data = copy.deepcopy(gt.dataset)
    images = {item['id']: item for item in data['images']}
    def proxy_area(ann):
        image = images[ann['image_id']]
        ratio = min(640 / image['width'], 640 / image['height'])
        return (max(ann['bbox'][2:4]) * ratio) ** 2
    for ann in data['annotations']:
        ann['area'] = proxy_area(ann)
    scaled_gt = COCO()
    scaled_gt.dataset = data
    scaled_gt.createIndex()
    scaled_dt = scaled_gt.loadRes(predictions)
    for ann in scaled_dt.dataset['annotations']:
        ann['area'] = proxy_area(ann)
    scaled_dt.createIndex()
    groups = []
    for name, low, high in (('Tiny', 0, 64), ('Small', 64, 256),
                            ('Medium', 256, 1024), ('Large', 1024, 1e12)):
        row = {'group': name, 'gt_count': sum(
            (low < ann['area'] <= high) if low else (ann['area'] <= high)
            for ann in data['annotations'] if not ann.get('iscrowd', 0))}
        row.update(coco_metrics(scaled_gt, scaled_dt,
                                (low + (1e-6 if low else 0), high)))
        groups.append(row)
    return groups


def matched_ious(gt, predictions):
    images = {item['id']: item for item in gt.dataset['images']}
    annotations = {}
    for ann in gt.dataset['annotations']:
        if not ann.get('iscrowd', 0):
            annotations.setdefault((ann['image_id'], ann['category_id']), []).append(ann)
    detections = {}
    for pred in predictions:
        detections.setdefault((pred['image_id'], pred['category_id']), []).append(pred)
    rows = []
    matched = {name: [] for name in ('Tiny', 'Small', 'Medium', 'Large')}
    for key, preds in detections.items():
        ground = annotations.get(key, [])
        used = set()
        for pred in sorted(preds, key=lambda item: item['score'], reverse=True):
            candidates = [(iou(pred['bbox'], ann['bbox']), index, ann)
                          for index, ann in enumerate(ground) if index not in used]
            best = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
            rows.append({'image_id': key[0], 'category_id': key[1], 'score': pred['score'],
                         'best_unmatched_iou': best[0]})
            if best[0] >= 0.5:
                used.add(best[1])
                matched[group_name(best[2], images[key[0]])].append(best[0])
    return rows, {name: sum(values) / len(values) if values else None
                  for name, values in matched.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='COCO GT JSON for the chosen split')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.annotations.is_file() or args.output_dir.exists():
        parser.error('weights/annotations must exist; output directory must be new')
    local_package()
    from pycocotools.coco import COCO
    from ultralytics import RTDETR
    from evaluate_rb_hrnet_laf import coco_metrics
    gt = COCO(str(args.annotations))
    result = RTDETR(str(args.weights)).val(data=str(DATA), split=args.split,
                                             device=args.device, batch=8, imgsz=640,
                                             max_det=300, save_json=True,
                                             project=str(args.output_dir.parent),
                                             name=args.output_dir.name, exist_ok=True)
    prediction_file = args.output_dir / 'predictions.json'
    predictions = json.loads(prediction_file.read_text(encoding='utf-8'))
    ids = {Path(image['file_name']).stem: image['id'] for image in gt.dataset['images']}
    if len(ids) != len(gt.dataset['images']):
        raise RuntimeError('annotation image stems are not unique')
    for pred in predictions:
        stem = str(pred['image_id'])
        if stem not in ids:
            raise RuntimeError(f'prediction image ID not found in annotations: {stem}')
        pred['image_id'] = ids[stem]
    prediction_file.write_text(json.dumps(predictions), encoding='utf-8')
    overall = coco_metrics(gt, gt.loadRes(predictions))
    groups = measure_groups(gt, predictions)
    rows, matched = matched_ious(gt, predictions)
    for group in groups:
        group['mean_matched_iou_at_50'] = matched[group['group']]
    overall['ultralytics'] = getattr(result, 'results_dict', {})
    (args.output_dir / 'evaluation_metrics.json').write_text(
        json.dumps(overall, indent=2), encoding='utf-8')
    with (args.output_dir / 'size_group_metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with (args.output_dir / 'localization_analysis.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else
                                ['image_id', 'category_id', 'score', 'best_unmatched_iou'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'overall': overall, 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
