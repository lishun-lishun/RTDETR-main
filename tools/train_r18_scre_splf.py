#!/usr/bin/env python3
"""Train one SCRE/SPLF ablation. Dry-run by default; --train starts training.

Formal runs require the real completed baseline's args.yaml so that all
optimizer, augmentation, and schedule settings are copied rather than guessed.
Protocol ``frozen`` is a separate, short diagnostic initialized from best.pt.
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml

from scre_splf_common import (DATA_CONFIG, MODEL_DIR, OUTPUT_ROOT, PROFILE_CONFIG, ROOT,
                              VARIANTS, main_layer_indices, read_baseline_args,
                              transfer_baseline, write_json, yaml_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--protocol", choices=("scratch", "frozen"), default="scratch")
    parser.add_argument("--baseline-args", type=Path, help="Actual baseline run args.yaml; required with --train")
    parser.add_argument("--baseline-weights", type=Path, help="Original R18 best.pt; required for frozen protocol")
    parser.add_argument("--diagnostic-epochs", type=int, default=10)
    parser.add_argument("--project-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train", action="store_true", help="Run tests and start training")
    return parser.parse_args()


def settings_from_baseline(record, args):
    from ultralytics.utils import DEFAULT_CFG_DICT

    settings = {k: v for k, v in record.items() if k in DEFAULT_CFG_DICT}
    for key in ("model", "data", "project", "name", "device", "resume", "save_dir", "cfg"):
        settings.pop(key, None)
    if args.protocol == "frozen":
        settings["epochs"] = args.diagnostic_epochs
        settings["freeze"] = main_layer_indices(args.model)
    else:
        settings["freeze"] = None
    settings.update(data=str(DATA_CONFIG), device=args.device,
                    project=str(args.project_root / args.protocol / args.model),
                    name="train", exist_ok=False, resume=False)
    return settings


def append_event(directory, message):
    with (directory / "execution_log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"[{dt.datetime.now().astimezone().isoformat()}] {message}\n")


def main():
    args = parse_args()
    cfg = yaml_path(args.model)
    output = args.project_root / args.protocol / args.model / "train"
    baseline_record = read_baseline_args(args.baseline_args) if args.baseline_args else None
    print(f"model={args.model} config={cfg}")
    print(f"protocol={args.protocol} output={output} device={args.device}")
    print(f"dataset={DATA_CONFIG}")
    print(f"baseline_args={args.baseline_args or '<required for --train>'}")
    print(f"baseline_weights={args.baseline_weights or '<required for frozen>'}")
    if baseline_record:
        print("baseline settings: " + ", ".join(f"{k}={baseline_record.get(k)}" for k in
              ("epochs", "batch", "gradient_accumulation", "imgsz", "optimizer", "lr0",
               "lrf", "weight_decay", "warmup_epochs", "seed", "amp", "pretrained")))
    if not args.train:
        print("Dry-run only. Pass --train to run preflight tests and start this one model.")
        return
    if "," in args.device or args.device.lower() in {"cpu", "mps"}:
        raise SystemExit("This preflighted launcher expects one CUDA device per experiment, e.g. --device 1")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required on the training server; CPU-only skipped tests are insufficient") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to start a run without the required CUDA preflight")
    if baseline_record is None:
        raise SystemExit("--baseline-args is required: local profile is not proof of the actual baseline run")
    if args.protocol == "frozen" and args.baseline_weights is None:
        raise SystemExit("--baseline-weights is required for frozen diagnostic")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {output}")
    if not DATA_CONFIG.is_file() or not cfg.is_file():
        raise SystemExit("Dataset or model YAML is missing")

    print("Running mandatory SCRE/SPLF preflight tests...")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                    "-p", "test_scre_splf.py", "-v"], cwd=ROOT, check=True)

    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils.torch_utils import de_parallel

    settings = settings_from_baseline(baseline_record, args)
    metadata = {"model": args.model, "config": str(cfg.resolve()), "protocol": args.protocol,
                "dataset": str(DATA_CONFIG.resolve()), "baseline_args": str(args.baseline_args.resolve()),
                "baseline_weights": str(args.baseline_weights.resolve()) if args.baseline_weights else None,
                "settings": settings, "auxiliary": yaml.safe_load(cfg.read_text(encoding="utf-8"))["scre_splf"],
                "started_at": dt.datetime.now().astimezone().isoformat()}

    class ExperimentTrainer(RTDETRTrainer):
        def __init__(self, *positional, **keyword):
            super().__init__(*positional, **keyword)
            write_json(self.save_dir / "training_config.json", metadata)
            append_event(self.save_dir, "Trainer initialized; preflight tests passed")
            if args.protocol == "frozen":
                def keep_main_eval(trainer):
                    original = de_parallel(trainer.model)
                    for index in main_layer_indices(args.model):
                        original.model[index].eval()
                self.add_callback("on_train_batch_start", keep_main_eval)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            transfer = None
            if args.protocol == "frozen":
                transfer = transfer_baseline(model, args.baseline_weights, args.model)
            write_json(self.save_dir / "model_summary.json", {
                "model": args.model,
                "layers": len(model.model),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters_before_freeze": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "baseline_transfer": transfer,
                "decoder_inputs": model.model[-1].f,
            })
            append_event(self.save_dir, f"Model constructed; transfer={transfer}")
            return model

    experiment = RTDETR(str(cfg))
    try:
        experiment.train(trainer=ExperimentTrainer, **settings)
    except Exception as exc:
        if output.exists():
            append_event(output, f"Training failed: {type(exc).__name__}: {exc}")
        raise
    if output.exists():
        append_event(output, "Training completed")
        print(f"best_weights={output / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
