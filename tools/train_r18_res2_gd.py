"""Train one Res2BasicBlock/LiteGDNeck RT-DETR-R18 experiment on one GPU."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import yaml

from res2_gd_common import CONFIGS, DATA, PROFILE, ROOT, RUNS, config_path, local_package


def settings_for_run(args, parser):
    profile = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    if args.baseline_args:
        if not args.baseline_args.is_file():
            parser.error('baseline args.yaml does not exist')
        baseline = yaml.safe_load(args.baseline_args.read_text(encoding='utf-8'))
        keys = ('epochs', 'imgsz', 'optimizer', 'lr0', 'lrf', 'weight_decay',
                'amp', 'pretrained')
        mismatches = {key: (baseline.get(key), profile[key]) for key in keys
                      if baseline.get(key) != profile[key]}
        accumulation = baseline.get('gradient_accumulation', 1)
        batch = baseline.get('batch', 0) * accumulation
        if baseline.get('warmup_epochs', 0) / accumulation != profile['warmup_epochs']:
            mismatches['warmup_optimizer_iterations'] = (
                baseline.get('warmup_epochs', 0) / accumulation,
                profile['warmup_epochs'])
        if mismatches or batch != 8:
            parser.error(f'baseline protocol differs: {mismatches}; effective batch={batch}')
        allowed = yaml.safe_load((ROOT / 'ultralytics/cfg/default.yaml').read_text(encoding='utf-8'))
        settings = {key: value for key, value in baseline.items() if key in allowed}
        settings['batch'] = profile['micro_batch']
        settings['gradient_accumulation'] = batch // profile['micro_batch']
        settings['warmup_epochs'] = (profile['warmup_epochs'] *
                                     settings['gradient_accumulation'])
    else:
        print('WARNING: original R18 args.yaml unavailable; historical protocol parity '
              'is unverified. Using the checked-in profile.', flush=True)
        batch = profile['batch']
        settings = {key: value for key, value in profile.items()
                    if key not in ('test_batch', 'micro_batch')}
        settings['gradient_accumulation'] = batch // profile['micro_batch']
        settings['warmup_epochs'] = (profile['warmup_epochs'] *
                                     settings['gradient_accumulation'])
        settings['batch'] = profile['micro_batch']
    for key in ('model', 'mode', 'project', 'name', 'device', 'resume',
                'save_dir', 'cfg', 'data'):
        settings.pop(key, None)
    settings.update(data=str(DATA), device=args.device,
                    project=str(args.output_root / args.model),
                    name=f'seed{args.seed}', exist_ok=True, seed=args.seed)
    return settings, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--device', required=True, help='One physical CUDA index')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=RUNS)
    parser.add_argument('--baseline-args', type=Path,
                        help='Completed original R18 args.yaml for exact protocol audit')
    parser.add_argument('--resume', type=Path, help='last.pt of this model and seed')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.device.isdecimal() or args.seed < 0:
        parser.error('device must be one CUDA index and seed must be nonnegative')
    output = args.output_root / args.model / f'seed{args.seed}'
    if args.resume and args.resume.resolve() != (output / 'weights/last.pt').resolve():
        parser.error('resume must point to this model/seed weights/last.pt')
    settings, batch = settings_for_run(args, parser)
    cfg = config_path(args.model)
    print(json.dumps({'model_yaml': str(cfg), 'output': str(output),
                      'effective_batch': batch, 'settings': settings,
                      'resume': str(args.resume) if args.resume else None},
                     indent=2, default=str), flush=True)
    if args.dry_run:
        return
    if output.exists() and not args.resume:
        parser.error(f'experiment output already exists: {output}')
    if args.resume and not args.resume.is_file():
        parser.error('resume checkpoint is missing')
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules import LiteGDNeck, Res2Conv
    from ultralytics.nn.tasks import RTDETRDetectionModel
    if not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count():
        parser.error('requested CUDA GPU is unavailable')
    if not cfg.is_file() or not DATA.is_file():
        parser.error('model or dataset YAML is missing')
    if not args.resume:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'),
                        '-p', 'test_res2basic_litegd.py', '-v'], cwd=ROOT, check=True)
        network = RTDETRDetectionModel(str(cfg), nc=1, verbose=False).eval()
        incoming = []
        hook = network.model[-1].register_forward_pre_hook(
            lambda _m, values: incoming.extend(tuple(x.shape) for x in values[0]))
        with torch.inference_mode():
            network(torch.zeros(1, 3, 128, 128))
        hook.remove()
        if incoming != [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)]:
            raise RuntimeError(f'decoder input mismatch: {incoming}')
        res2_count = sum(isinstance(m, Res2Conv) for m in network.modules())
        gd_count = sum(isinstance(m, LiteGDNeck) for m in network.modules())
        expected_res2 = 2 if args.model in {'res2basic_both', 'res2basic_stage34'} else (
            1 if args.model.startswith('res2basic') else 0)
        expected_gd = int('litegd' in args.model)
        if (res2_count, gd_count) != (expected_res2, expected_gd):
            raise RuntimeError(f'expected Res2/GD {(expected_res2, expected_gd)}, '
                               f'got {(res2_count, gd_count)}')
        summary = {'layers': len(network.model),
                   'parameters': sum(p.numel() for p in network.parameters()),
                   'decoder_sources': network.model[-1].f,
                   'decoder_shapes_at_128': incoming,
                   'res2_blocks': res2_count, 'gd_necks': gd_count}
        del network
        model = RTDETR(str(cfg))
    else:
        model = RTDETR(str(args.resume))
        settings['resume'] = str(args.resume)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output.mkdir()
        (output / 'training_config.json').write_text(json.dumps({
            'model': args.model, 'yaml': str(cfg.resolve()), 'dataset': str(DATA.resolve()),
            'baseline_args': str(args.baseline_args) if args.baseline_args else None,
            'effective_batch': batch, 'settings': settings,
            'started_at': dt.datetime.now().astimezone().isoformat()}, indent=2), encoding='utf-8')
        (output / 'model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log_path = output / 'execution_log.txt'
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'[{dt.datetime.now().astimezone().isoformat()}] training started\n')
    try:
        model.train(**settings)
    except Exception as exc:
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'[{dt.datetime.now().astimezone().isoformat()}] '
                      f'failed: {type(exc).__name__}: {exc}\n')
        raise
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'[{dt.datetime.now().astimezone().isoformat()}] training complete\n')
    results_csv = output / 'results.csv'
    if not results_csv.is_file():
        raise RuntimeError(f'trainer did not write per-epoch results: {results_csv}')
    with results_csv.open(newline='', encoding='utf-8') as file:
        rows = [{key.strip(): value for key, value in row.items()}
                for row in csv.DictReader(file)]
    if not rows or not any(key.startswith('lr/pg') for key in rows[0]):
        raise RuntimeError('results.csv has no actual learning-rate columns')
    (output / 'epoch_history.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
