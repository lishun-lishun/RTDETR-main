"""Train one R18 SA-Rep/SP-ASFF variant for 150 epochs on a single GPU."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import yaml

from sarep_spasff_common import CONFIGS, DATA, PROFILE, ROOT, RUNS, config_path, local_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--device', required=True, help='One physical CUDA device index, e.g. 0')
    parser.add_argument('--output-root', type=Path, default=RUNS)
    parser.add_argument('--baseline-args', type=Path,
                        help='Completed original R18 args.yaml for protocol verification')
    parser.add_argument('--resume', type=Path, help='last.pt from this same experiment')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.device.isdecimal():
        parser.error('--device must name exactly one physical CUDA GPU')
    output = args.output_root / args.model
    if args.resume and args.resume.resolve() != (output / 'weights/last.pt').resolve():
        parser.error('resume must point to this experiment\'s weights/last.pt')
    profile = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    if args.baseline_args:
        if not args.baseline_args.is_file():
            parser.error('baseline args.yaml does not exist')
        baseline = yaml.safe_load(args.baseline_args.read_text(encoding='utf-8'))
        fields = ('epochs', 'imgsz', 'optimizer', 'lr0', 'lrf', 'weight_decay',
                  'seed', 'amp', 'pretrained')
        expected = {key: profile[key] for key in fields}
        mismatch = {key: (baseline.get(key), value) for key, value in expected.items()
                    if baseline.get(key) != value}
        accumulation = baseline.get('gradient_accumulation', 1)
        actual_batch = baseline.get('batch', 0) * accumulation
        warmup_iterations = baseline.get('warmup_epochs', 0) / accumulation
        if warmup_iterations != profile['warmup_epochs']:
            mismatch['warmup_optimizer_iterations'] = (warmup_iterations, profile['warmup_epochs'])
        if mismatch or actual_batch != 8:
            parser.error(f'baseline protocol differs: {mismatch}; effective batch={actual_batch}')
        allowed = yaml.safe_load((ROOT / 'ultralytics/cfg/default.yaml').read_text(encoding='utf-8'))
        settings = {key: value for key, value in baseline.items() if key in allowed}
        settings['gradient_accumulation'] = accumulation
    else:
        print('WARNING: baseline args.yaml unavailable; using checked-in profile, '
              'so historical run equivalence is unverified.', flush=True)
        actual_batch = profile['batch']
        settings = {key: value for key, value in profile.items()
                    if key not in ('test_batch', 'micro_batch')}
        settings['gradient_accumulation'] = profile['batch'] // profile['micro_batch']
        settings['warmup_epochs'] = profile['warmup_epochs'] * settings['gradient_accumulation']
        settings['batch'] = profile['micro_batch']
    for key in ('model', 'mode', 'project', 'name', 'device', 'resume', 'save_dir', 'cfg', 'data'):
        settings.pop(key, None)
    settings.update(data=str(DATA), device=args.device, project=str(args.output_root),
                    name=args.model, exist_ok=True)
    print(json.dumps({'config': str(config_path(args.model)), 'output': str(output),
                      'settings': settings, 'resume': str(args.resume) if args.resume else None},
                     indent=2, default=str), flush=True)
    if args.dry_run:
        return
    if output.exists() and not args.resume:
        parser.error(f'output already exists: {output}')
    if args.resume and not args.resume.is_file():
        parser.error('resume checkpoint does not exist')
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.modules.sarep_spasff import SARepConv
    from ultralytics.nn.tasks import RTDETRDetectionModel
    if not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count():
        parser.error('requested CUDA GPU is unavailable')
    cfg = config_path(args.model)
    if not cfg.is_file() or not DATA.is_file():
        parser.error('model or dataset YAML missing')
    if not args.resume:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'),
                        '-p', 'test_sarep_spasff.py', '-v'], cwd=ROOT, check=True)
        network = RTDETRDetectionModel(str(cfg), nc=1, verbose=False).eval()
        incoming = []
        handle = network.model[-1].register_forward_pre_hook(
            lambda _m, inputs: incoming.extend(tuple(x.shape) for x in inputs[0]))
        with torch.inference_mode():
            network(torch.zeros(1, 3, 128, 128))
        handle.remove()
        if incoming != [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)]:
            raise RuntimeError(f'decoder input shape mismatch: {incoming}')
        expected_blocks = {'sarep': 1, 'sarep_both': 2, 'sarep_stage34': 2,
                           'sarep_spasff': 1}.get(args.model, 0)
        actual_blocks = sum(isinstance(m, SARepConv) for m in network.modules())
        if actual_blocks != expected_blocks:
            raise RuntimeError(f'expected {expected_blocks} SARepConv, got {actual_blocks}')
        summary = {'model': args.model, 'layers': len(network.model),
                   'parameters': sum(p.numel() for p in network.parameters()),
                   'decoder_sources': network.model[-1].f,
                   'decoder_shapes_at_128': incoming, 'sarep_blocks': actual_blocks}
        del network
        model = RTDETR(str(cfg))
    else:
        model = RTDETR(str(args.resume))
        settings['resume'] = str(args.resume)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output.mkdir()
        (output / 'training_config.json').write_text(json.dumps({
            'model': args.model, 'model_yaml': str(cfg.resolve()),
            'dataset': str(DATA.resolve()), 'baseline_args': str(args.baseline_args) if args.baseline_args else None,
            'effective_batch': actual_batch, 'settings': settings,
            'started_at': dt.datetime.now().astimezone().isoformat()}, indent=2), encoding='utf-8')
        (output / 'model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (output / 'checkpoint_audit.json').write_text(json.dumps({
            'initialization': 'random YAML construction', 'pretrained': False,
            'baseline_checkpoint_loaded': False,
            'baseline_args_verified': args.baseline_args is not None}, indent=2), encoding='utf-8')
    with (output / 'execution_log.txt').open('a', encoding='utf-8') as log:
        log.write(f'[{dt.datetime.now().astimezone().isoformat()}] training started\n')
    try:
        model.train(**settings)
    except Exception as exc:
        with (output / 'execution_log.txt').open('a', encoding='utf-8') as log:
            log.write(f'[{dt.datetime.now().astimezone().isoformat()}] '
                      f'training failed: {type(exc).__name__}: {exc}\n')
        raise
    with (output / 'execution_log.txt').open('a', encoding='utf-8') as log:
        log.write(f'[{dt.datetime.now().astimezone().isoformat()}] training completed\n')


if __name__ == '__main__':
    main()
