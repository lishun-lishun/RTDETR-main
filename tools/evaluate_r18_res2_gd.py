"""COCO evaluation and custom 640-input size groups for Res2/GD checkpoints."""
import argparse
import csv
import json
from pathlib import Path

from res2_gd_common import CONFIGS, DATA, local_package
from evaluate_r18_sarep_spasff import matched_ious, measure_groups
from evaluate_rb_hrnet_laf import coco_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.annotations.is_file() or args.output_dir.exists():
        parser.error('weights and annotations must exist; output directory must be new')
    local_package()
    from pycocotools.coco import COCO
    from ultralytics import RTDETR

    gt = COCO(str(args.annotations))
    result = RTDETR(str(args.weights)).val(
        data=str(DATA), split=args.split, device=args.device, batch=8,
        imgsz=640, max_det=300, save_json=True,
        project=str(args.output_dir.parent), name=args.output_dir.name,
        exist_ok=True)
    prediction_file = args.output_dir / 'predictions.json'
    predictions = json.loads(prediction_file.read_text(encoding='utf-8'))
    ids = {Path(image['file_name']).stem: image['id'] for image in gt.dataset['images']}
    if len(ids) != len(gt.dataset['images']):
        raise RuntimeError('COCO annotation image stems must be unique')
    for prediction in predictions:
        stem = str(prediction['image_id'])
        if stem not in ids:
            raise RuntimeError(f'unknown prediction image ID: {stem}')
        prediction['image_id'] = ids[stem]
    prediction_file.write_text(json.dumps(predictions), encoding='utf-8')
    overall = coco_metrics(gt, gt.loadRes(predictions))
    groups = measure_groups(gt, predictions)
    rows, matched = matched_ious(gt, predictions)
    for group in groups:
        group['mean_matched_iou_at_50'] = matched[group['group']]
    native = getattr(result, 'results_dict', {})
    overall.update(model=args.model, weights=str(args.weights), split=args.split,
                   input_size=640, max_detections=300,
                   size_group_definition='maximum bbox side after 640 letterbox',
                   precision=native.get('metrics/precision(B)'),
                   recall=native.get('metrics/recall(B)'),
                   ultralytics=native)
    (args.output_dir / 'evaluation_metrics.json').write_text(
        json.dumps(overall, indent=2), encoding='utf-8')
    for filename, data, columns in (
        ('size_group_metrics.csv', groups, list(groups[0])),
        ('localization_analysis.csv', rows,
         ['image_id', 'category_id', 'score', 'best_unmatched_iou'])):
        with (args.output_dir / filename).open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({'overall': overall, 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
