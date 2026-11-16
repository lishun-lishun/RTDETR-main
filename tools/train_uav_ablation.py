"""Train one R18 UAV ablation with the repository's checked-in R18 protocol.

Dry run by default. Add --execute to start training on the selected device.
All runs use the same profile unless the caller explicitly changes an argument.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml

from r18_targeted_common import DATA, MODEL_DIR, ROOT, git_state, local_package, protocol_settings, write_json


CONFIGS = {
    "e0_baseline": "rtdetr-r18.yaml",
    "e1_queryonly": "rtdetr_r18_uav_queryonly.yaml",
    "e2_evidence_refine": "rtdetr_r18_uav_evidence_refine.yaml",
    "e3_quality": "rtdetr_r18_uav_quality.yaml",
    "e4_loss_only": "rtdetr_r18_uav_loss_only.yaml",
    "e5_all": "rtdetr_r18_uav_all.yaml",
    "e6_p2_queryonly": "rtdetr_r18_uav_p2_queryonly.yaml",
    "e7_p2_evidence": "rtdetr_r18_uav_p2_evidence.yaml",
    "e8_p2_all": "rtdetr_r18_uav_p2_all.yaml",
    "tinyv2_off_control": "rtdetr_r18_tinyv2_off_control.yaml",
    "tinyv2_m1_peak": "rtdetr_r18_tinyv2_m1_peak.yaml",
    "tinyv2_m2_dn": "rtdetr_r18_tinyv2_m2_dn.yaml",
    "tinyv2_m3_match": "rtdetr_r18_tinyv2_m3_match.yaml",
    "tinyv2_m4_nwd": "rtdetr_r18_tinyv2_m4_nwd.yaml",
    "tinyv2_m1_m3": "rtdetr_r18_tinyv2_m1_m3.yaml",
    "tinyv2_m2_m4": "rtdetr_r18_tinyv2_m2_m4.yaml",
    "tinyv2_all4": "rtdetr_r18_tinyv2_all4.yaml",
}


def validate_config(name: str) -> Path:
    """Reject a changed backbone or an accidental four-level Transformer input."""
    path = MODEL_DIR / CONFIGS[name]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load((MODEL_DIR / CONFIGS["e0_baseline"]).read_text(encoding="utf-8"))
    if cfg["backbone"] != base["backbone"] or cfg["head"][:-1] != base["head"][:-1]:
        raise ValueError(f"{name}: R18 backbone or neck differs from the baseline")
    head = cfg["head"][-1]
    if name.startswith("tinyv2_"):
        if head[0] != [19, 22, 25] or head[2] != "TinyUAVDecoder" or head[3][2] != 300:
            raise ValueError(f"{name}: expected P3/P4/P5 and exactly 300 TinyV2 queries")
        if not isinstance(cfg.get("tiny_uav"), dict):
            raise ValueError(f"{name}: missing tiny_uav settings")
    elif name in ("e0_baseline", "e4_loss_only"):
        if head != base["head"][-1]:
            raise ValueError(f"{name}: expected the original RTDETRDecoder")
    else:
        p2 = name.startswith(("e6_", "e7_", "e8_"))
        expected = [4, 19, 22, 25] if p2 else [19, 22, 25]
        if head[0] != expected or head[2] != "UAVResearchDecoder":
            raise ValueError(f"{name}: unexpected Decoder feature sources")
        if bool(head[3][-1].get("p2_source", False)) != p2:
            raise ValueError(f"{name}: P2 source flag disagrees with its routes")
        if head[3][2] != 300:  # nc, hidden_dim, query_count, ...
            raise ValueError(f"{name}: query count must remain 300")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CONFIGS)
    parser.add_argument("--device", required=True, help="CUDA index (e.g. 0 or 1), or cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8, help="effective batch size")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-args", type=Path, help="historical R18 args.yaml for protocol comparison")
    parser.add_argument("--init-weights", type=Path, help="optional identical R18 checkpoint for every run")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "uav_ablation")
    parser.add_argument("--execute", action="store_true", help="start training; otherwise print the plan")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0 or args.batch < 1 or args.micro_batch < 1:
        parser.error("epochs and batch sizes must be positive; seed must be nonnegative")
    try:
        config = validate_config(args.model)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    settings, protocol = protocol_settings(args, parser)
    output = args.output_root / args.model / f"seed{args.seed}"
    payload = {
        "model": args.model, "yaml": str(config.resolve()), "data": str(DATA.resolve()),
        "output": str(output.resolve()), "init_weights": str(args.init_weights) if args.init_weights else None,
        "settings": settings, "protocol": protocol, "git": git_state(),
    }
    print(json.dumps(payload, indent=2, default=str), flush=True)
    if not args.execute:
        return
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not DATA.is_file():
        parser.error(f"dataset YAML is missing: {DATA}")
    if args.init_weights and not args.init_weights.is_file():
        parser.error(f"initial weights are missing: {args.init_weights}")
    local_package()
    import torch
    from ultralytics import RTDETR

    if args.device != "cpu":
        if not args.device.isdecimal() or not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count():
            parser.error(f"CUDA device {args.device!r} is unavailable")
    # Model construction precedes trainer seeding; seed here so all ablations
    # start from identical R18 parameters when no checkpoint is supplied.
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = RTDETR(str(config))
    if args.init_weights:
        model.load(str(args.init_weights))
    write_json(output / "training_config.json", {**payload, "started_at": dt.datetime.now().astimezone().isoformat()})
    model.train(**settings)


if __name__ == "__main__":
    main()
