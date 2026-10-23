"""Validate a checkpoint and compute COCO AP by 640px maximum-edge groups."""
import argparse
import copy
import csv
import json
from pathlib import Path

import yaml

from train_rb_hrnet_laf import CONFIGS, ROOT


def ap_recall(evaluator):
    precision = evaluator.eval['precision']
    recall = evaluator.eval['recall']
    result = {}
    for label, indices in (('50', [0]), ('75', [5]), ('50_95', range(10))):
        p = precision[indices, :, :, 0, -1]
        r = recall[indices, :, 0, -1]
        result['AP' + label] = float(p[p >= 0].mean()) if (p >= 0).any() else None
        result['Recall' + label] = float(r[r >= 0].mean()) if (r >= 0).any() else None
    return result


def coco_metrics(gt, dt, area=None):
    from pycocotools.cocoeval import COCOeval
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.imgIds = sorted(gt.getImgIds())
    evaluator.params.maxDets = [1, 10, 300]
    if area:
        evaluator.params.areaRng = [list(area)]
        evaluator.params.areaRngLbl = ['custom']
    evaluator.evaluate()
    evaluator.accumulate()
    return ap_recall(evaluator)


def measure_groups(gt, predictions):
    from pycocotools.coco import COCO
    data = copy.deepcopy(gt.dataset)
    images = {item['id']: item for item in data['images']}
    def area(ann):
        image = images[ann['image_id']]
        w, h = ann['bbox'][2:4]
        return max(w * 640 / image['width'], h * 640 / image['height']) ** 2
    for ann in data['annotations']:
        ann['area'] = area(ann)
    resized_gt = COCO()
    resized_gt.dataset = data
    resized_gt.createIndex()
    resized_dt = resized_gt.loadRes(predictions)
    for ann in resized_dt.dataset['annotations']:
        ann['area'] = area(ann)
    resized_dt.createIndex()
    rows = []
    for name, low, high in (('Tiny', 0, 64), ('Small', 64, 256),
                            ('Medium', 256, 1024), ('Large', 1024, 1e12)):
        row = {'group': name,
               'gt_count': sum((low < ann['area'] <= high) if low
                               else (ann['area'] <= high)
                               for ann in data['annotations'])}
        row.update(coco_metrics(resized_gt, resized_dt,
                                (low + (1e-6 if low else 0), high)))
        rows.append(row)
    return rows


def localization_rows(gt, predictions):
    """Per-detection score/maximum-IoU diagnostics; not an AP substitute."""
    by_image = {}
    for ann in gt.dataset['annotations']:
        if not ann.get('iscrowd', 0):
            by_image.setdefault((ann['image_id'], ann['category_id']), []).append(ann)
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
        intersection = overlap_w * overlap_h
        union = max(0, aw * ah) + max(0, bw * bh) - intersection
        return intersection / union if union > 0 else 0.0
    for pred in predictions:
        candidates = by_image.get((pred['image_id'], pred['category_id']), [])
        yield {'image_id': pred['image_id'],
               'category_id': pred['category_id'],
               'score': pred['score'],
               'max_iou_same_class': max(
                   (iou(pred['bbox'], ann['bbox']) for ann in candidates),
                   default=0.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=CONFIGS)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error('weights file does not exist')
    if args.output_dir.exists():
        parser.error('output directory already exists')
    from pycocotools.coco import COCO
    from ultralytics import RTDETR

    data_yaml = ROOT / 'dataset/DUT-Anti-UAV.yaml'
    dataset = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    gt = COCO(str(Path(dataset['path']) / 'labels' / (args.split + '.json')))
    model = RTDETR(str(args.weights))
    model.val(data=str(data_yaml), split=args.split, device=args.device,
              batch=8, imgsz=640, save_json=True,
              project=str(args.output_dir.parent), name=args.output_dir.name)
    prediction_file = args.output_dir / 'predictions.json'
    predictions = json.loads(prediction_file.read_text(encoding='utf-8'))
    ids = {Path(image['file_name']).stem: image['id']
           for image in gt.dataset['images']}
    for pred in predictions:
        pred['image_id'] = ids[str(pred['image_id'])]
    prediction_file.write_text(json.dumps(predictions), encoding='utf-8')
    overall = coco_metrics(gt, gt.loadRes(predictions))
    groups = measure_groups(gt, predictions)
    (args.output_dir / 'evaluation_metrics.json').write_text(
        json.dumps(overall, indent=2), encoding='utf-8')
    with (args.output_dir / 'size_group_metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with (args.output_dir / 'localization_analysis.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=(
            'image_id', 'category_id', 'score', 'max_iou_same_class'))
        writer.writeheader()
        writer.writerows(localization_rows(gt, predictions))
    print(json.dumps({'overall': overall, 'size_groups': groups}, indent=2))


if __name__ == '__main__':
    main()
