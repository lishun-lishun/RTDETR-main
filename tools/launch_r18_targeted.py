"""Queue targeted R09--R16 runs across explicitly assigned GPUs.

Dry-run is the default. ``--execute`` creates isolated logs, sets
``CUDA_VISIBLE_DEVICES`` and records code/config hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from r18_targeted_common import CONFIGS, ROOT, model_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--devices", "--gpus", required=True, help="comma-separated CUDA IDs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "r18_targeted")
    parser.add_argument("--batch", type=int, default=8); parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-parallel", type=int); parser.add_argument("--resume-models", default="")
    parser.add_argument("--baseline-args", type=Path); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try: names = model_names(args.models)
    except ValueError as exc: parser.error(str(exc))
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices: parser.error("at least one device is required")
    if len(devices) < len(names): devices = [devices[index % len(devices)] for index in range(len(names))]
    resume = set(model_names(args.resume_models)) if args.resume_models else set()
    args.max_parallel = args.max_parallel or len(set(devices))
    if args.max_parallel < 1: parser.error("--max-parallel must be positive")
    jobs = []
    for index, name in enumerate(names):
        device, run_dir = devices[index], args.output_root / name / f"seed{args.seed}"
        # The child sees only its assigned physical GPU through
        # CUDA_VISIBLE_DEVICES, therefore its local CUDA index is always 0.
        cmd = [sys.executable, str(ROOT / "tools" / "train_r18_targeted.py"), "--model", name, "--device", "0",
               "--output-root", str(args.output_root), "--batch", str(args.batch), "--micro-batch", str(args.micro_batch),
               "--epochs", str(args.epochs), "--seed", str(args.seed), "--execute"]
        if args.baseline_args: cmd += ["--baseline-args", str(args.baseline_args)]
        if name in resume: cmd += ["--resume", str(run_dir / "weights" / "last.pt")]
        jobs.append({"model": name, "device": "0", "physical_device": device, "run_dir": str(run_dir), "command": cmd,
                     "config_sha256": sha256(ROOT / "ultralytics/cfg/models/rt-detr" / CONFIGS[name]),
                     "code_sha256": sha256(ROOT / "tools/train_r18_targeted.py")})
    manifest = args.output_root / f"targeted_plan_seed{args.seed}.json"; payload = {"models": names, "jobs": jobs, "execute": args.execute, "created_at": time.time()}
    print(json.dumps(payload, indent=2))
    if not args.execute: return
    args.output_root.mkdir(parents=True, exist_ok=True); manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pending, active, completed = list(jobs), {}, []
    try:
        while pending or active:
            free = [d for d in sorted(set(devices)) if d not in active]
            while pending and free and len(active) < args.max_parallel:
                job, device = pending.pop(0), free.pop(0); run_dir = Path(job["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "launcher_stdout.log"; log = log_path.open("a", encoding="utf-8"); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(device)
                proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job.update({"status": "running", "pid": proc.pid, "log": str(log_path)}); active[device] = (job, proc, log)
            if not active: break
            time.sleep(2)
            for device, (job, proc, log) in list(active.items()):
                if proc.poll() is None: continue
                log.close(); text = Path(job["log"]).read_text(encoding="utf-8", errors="ignore"); oom = "out of memory" in text.lower()
                job.update({"status": "completed" if proc.returncode == 0 else "failed", "return_code": proc.returncode, "oom": oom})
                if oom: (Path(job["run_dir"]) / "OOM.txt").write_text(text[-4000:], encoding="utf-8")
                completed.append(job); del active[device]
    finally:
        payload["jobs"] = completed + pending + [value[0] for value in active.values()]; manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if any(item.get("status") == "failed" for item in completed): raise SystemExit("one or more targeted jobs failed")


if __name__ == "__main__": main()
