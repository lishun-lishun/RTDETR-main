"""Train one CSP R18 stage variant on a single GPU for the baseline protocol."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

from csp_context_common import (CONFIGS, DATA, MODEL_DIR, ROOT, RUNS,
                                local_package, settings_for_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--device', required=True, help='One visible CUDA index')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=RUNS)
    parser.add_argument('--baseline-args', type=Path,
                        help='Historical baseline args.yaml for protocol audit')
    parser.add_argument('--resume', type=Path, help='This run weights/last.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.device.isdecimal() or args.seed < 0:
        parser.error('device must be one visible CUDA index and seed nonnegative')
    output = args.output_root / args.model / f'seed{args.seed}'
    config = MODEL_DIR / CONFIGS[args.model]
    if args.resume and args.resume.resolve() != (output / 'weights/last.pt').resolve():
        parser.error('resume must be this model/seed weights/last.pt')
    settings, effective_batch = settings_for_run(args, parser)
    print(json.dumps({'config': str(config), 'output': str(output),
                      'effective_batch': effective_batch, 'settings': settings},
                     indent=2), flush=True)
    if args.dry_run:
        return
    if output.exists() and not args.resume:
        parser.error(f'experiment output already exists: {output}')
    if args.resume and not args.resume.is_file():
        parser.error('resume checkpoint is missing')
    if not config.is_file() or not DATA.is_file():
        parser.error('model or dataset YAML is missing')
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules import CSPContextStage, CSPResidualStage
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count():
        parser.error('requested visible CUDA index is unavailable')
    if not args.resume:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                        '-s', str(ROOT / 'tests'), '-p', 'test_csp_context_r18.py', '-v'],
                       cwd=ROOT, check=True)
        network = RTDETRDetectionModel(str(config), nc=1, verbose=False).eval()
        found = {}
        hooks = []
        for index in (5, 6, 7):
            hooks.append(network.model[index].register_forward_hook(
                lambda _m, _i, o, key=index: found.__setitem__(key, tuple(o.shape))))
        hooks.append(network.model[-1].register_forward_pre_hook(
            lambda _m, values: found.__setitem__(
                'decoder', [tuple(x.shape) for x in values[0]])))
        with torch.inference_mode():
            network(torch.zeros(1, 3, 128, 128))
        for hook in hooks:
            hook.remove()
        expected = {5: (1, 128, 16, 16), 6: (1, 256, 8, 8),
                    7: (1, 512, 4, 4), 'decoder': [(1, 256, 16, 16),
                                                  (1, 256, 8, 8), (1, 256, 4, 4)]}
        if found != expected or network.model[-1].f != [19, 22, 25]:
            raise RuntimeError(f'backbone/decoder preflight mismatch: {found}')
        has3 = isinstance(network.model[5], CSPResidualStage)
        has4 = isinstance(network.model[6], CSPContextStage)
        if (has3, has4) != (args.model in ('stage3', 'combined', 'stage3_ratio05'),
                            args.model in ('stage4', 'combined', 'stage4_k3')):
            raise RuntimeError('selected CSP stages do not match model name')
        summary = {'parameters': sum(p.numel() for p in network.parameters()),
                   'trainable_parameters': sum(p.numel() for p in network.parameters()
                                               if p.requires_grad),
                   'layers': len(network.model), 'stage3_csp': has3,
                   'stage4_csp': has4, 'shapes_at_128': found}
        del network
        model = RTDETR(str(config))
    else:
        model = RTDETR(str(args.resume))
        settings['resume'] = str(args.resume)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output.mkdir()
        (output / 'training_config.json').write_text(json.dumps({
            'model': args.model, 'yaml': str(config.resolve()),
            'dataset': str(DATA.resolve()),
            'baseline_args': str(args.baseline_args) if args.baseline_args else None,
            'effective_batch': effective_batch, 'settings': settings,
            'started_at': dt.datetime.now().astimezone().isoformat()},
            indent=2), encoding='utf-8')
        (output / 'model_summary.json').write_text(json.dumps(summary, indent=2),
                                                   encoding='utf-8')
    log_file = output / 'execution_log.txt'
    with log_file.open('a', encoding='utf-8') as file:
        file.write(f'[{dt.datetime.now().astimezone().isoformat()}] training started\n')
    try:
        model.train(**settings)
    except Exception as exc:
        with log_file.open('a', encoding='utf-8') as file:
            file.write(f'[{dt.datetime.now().astimezone().isoformat()}] '
                       f'failed: {type(exc).__name__}: {exc}\n')
        raise
    with log_file.open('a', encoding='utf-8') as file:
        file.write(f'[{dt.datetime.now().astimezone().isoformat()}] training complete\n')
    results_file = output / 'results.csv'
    if not results_file.is_file():
        raise RuntimeError(f'trainer did not save per-epoch results: {results_file}')
    with results_file.open(newline='', encoding='utf-8') as file:
        rows = [{key.strip(): value for key, value in row.items()}
                for row in csv.DictReader(file)]
    if not rows or not any(key.startswith('lr/pg') for key in rows[0]):
        raise RuntimeError('results.csv has no actual learning-rate columns')
    (output / 'epoch_history.json').write_text(json.dumps(rows, indent=2),
                                               encoding='utf-8')


if __name__ == '__main__':
    main()
