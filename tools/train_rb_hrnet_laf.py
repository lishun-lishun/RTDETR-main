"""Train the new Ultralytics RT-DETR variants in this RTDETR-main tree."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Direct execution sets sys.path[0] to tools/, which can otherwise select an
# older Ultralytics installation from site-packages instead of this checkout.
sys.path.insert(0, str(ROOT))
CONFIGS = {
    'hrnet_w18': 'rtdetr_hrnet_w18.yaml',
    'rb_hrnet_a': 'rtdetr_rb_hrnet_a.yaml',
    'rb_hrnet_b': 'rtdetr_rb_hrnet_b.yaml',
    'hrnet_laf0': 'rtdetr_hrnet_laf0.yaml',
    'hrnet_laf1': 'rtdetr_hrnet_laf1.yaml',
    'rb_hrnet_laf1': 'rtdetr_rb_hrnet_laf1.yaml',
}


def preflight(config):
    """Fail before training if the YAML graph or decoder connection is wrong."""
    import torch
    import ultralytics
    from ultralytics import RTDETR

    local_package = (ROOT / 'ultralytics').resolve()
    loaded_package = Path(ultralytics.__file__).resolve().parent
    if loaded_package != local_package:
        raise RuntimeError(
            f'Expected Ultralytics from {local_package}, got {loaded_package}')
    network = RTDETR(str(config)).model.eval()
    incoming = []
    hook = network.model[-1].register_forward_pre_hook(
        lambda _module, inputs: incoming.extend(tuple(x.shape) for x in inputs[0]))
    with torch.inference_mode():
        output = network(torch.zeros(1, 3, 128, 128))
    hook.remove()
    if incoming != [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)]:
        raise RuntimeError(f'Unexpected decoder inputs: {incoming}')
    if output[0].shape[1] != 300:
        raise RuntimeError(f'Unexpected decoder query count: {output[0].shape}')
    print('Preflight passed: P3/P4/P5 256 channels, 300 queries.', flush=True)
    del network
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=CONFIGS)
    parser.add_argument('--device', required=True,
                        help='Single physical CUDA device index, e.g. 1')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.device.isdecimal():
        parser.error('--device must be one GPU index')
    if args.model == 'rb_hrnet_laf1':
        print('Joint variant: run after reviewing independent backbone/neck validation.')
    cmd = [sys.executable, '-u', str(ROOT / 'train_dut_anti_uav.py'),
           '--mode', 'train', '--rt-detr-config', CONFIGS[args.model],
           '--device', args.device, '--epochs', '150',
           '--batch', '8', '--micro-batch', '8',
           '--train-project', str(ROOT / 'runs/rb-hrnet-laf/train')]
    print(' '.join(cmd), flush=True)
    if not args.dry_run:
        config = ROOT / 'ultralytics/cfg/models/rt-detr' / CONFIGS[args.model]
        preflight(config)
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            filter(None, (str(ROOT), env.get('PYTHONPATH'))))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
