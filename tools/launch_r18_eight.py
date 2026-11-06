"""Dry-run by default, then schedule independent R18 variants across GPUs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from r18_variants_common import CONFIGS, ROOT, RUNS, git_state, model_names, write_json


def gpu_inventory():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
        return [line for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(model, args):
    command = [sys.executable, str(ROOT / "tools" / "train_r18_variants.py"),
               "--model", model, "--device", "0", "--seed", str(args.seed),
               "--output-root", str(args.output_root), "--skip-tests"]
    if args.baseline_args:
        command.extend(("--baseline-args", str(args.baseline_args)))
    if args.batch is not None:
        command.extend(("--batch", str(args.batch)))
    if args.micro_batch is not None:
        command.extend(("--micro-batch", str(args.micro_batch)))
    if model in args.resume_models:
        checkpoint = args.output_root / model / f"seed{args.seed}" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"resume requested for {model}, but checkpoint is missing: {checkpoint}")
        command.extend(("--resume", str(checkpoint)))
    return command


def stop_model(model, output_root, seed):
    pid_path = output_root / model / f"seed{seed}" / "launcher.pid"
    if not pid_path.is_file():
        raise SystemExit(f"no launcher PID file for {model}: {pid_path}")
    pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"process {pid} is already stopped")
    else:
        print(f"sent SIGTERM to {model} process {pid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--gpus", required=True, help="physical GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--output-root", type=Path, default=RUNS)
    parser.add_argument("--baseline-args", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--resume-models", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-model", choices=CONFIGS)
    args = parser.parse_args()
    if args.stop_model:
        stop_model(args.stop_model, args.output_root, args.seed)
        return
    try:
        names = model_names(args.models)
        args.resume_models = set(model_names(args.resume_models)) if args.resume_models else set()
        gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(str(exc))
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain at least one unique integer")
    max_parallel = args.max_parallel or len(gpus)
    if max_parallel < 1 or max_parallel > len(gpus):
        parser.error("--max-parallel must be between 1 and the number of GPUs")
    inventory = gpu_inventory()
    if args.execute and not inventory:
        parser.error("--execute requires nvidia-smi and at least one visible NVIDIA GPU")
    if args.execute and any(gpu >= len(inventory) for gpu in gpus):
        parser.error(f"requested GPU IDs {gpus} exceed visible inventory of {len(inventory)}")
    for model in names:
        run_dir = args.output_root / model / f"seed{args.seed}"
        if run_dir.exists() and model not in args.resume_models:
            parser.error(f"experiment output already exists for {model}: {run_dir}")

    jobs = []
    for index, model in enumerate(names):
        gpu = gpus[index % len(gpus)]
        jobs.append({"model": model, "gpu": gpu, "command": command_for(model, args),
                     "env": {"CUDA_VISIBLE_DEVICES": str(gpu)},
                     "run_dir": str((args.output_root / model / f"seed{args.seed}").resolve())})
    manifest = {"created_at": dt.datetime.now().astimezone().isoformat(), "models": names,
                "gpus": gpus, "max_parallel": max_parallel, "seed": args.seed,
                "execute": args.execute, "git": git_state(), "gpu_inventory": inventory,
                "configs": {model: {"path": str((ROOT / "ultralytics/cfg/models/rt-detr" / CONFIGS[model]).resolve()),
                                    "sha256": file_hash(ROOT / "ultralytics/cfg/models/rt-detr" / CONFIGS[model])}
                            for model in names}, "jobs": jobs}
    manifest_path = args.output_root / f"sweep_seed{args.seed}_plan.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.execute:
        print(f"Dry run only. Review {manifest_path}, then rerun with --execute.")
        return

    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                    "-p", "test_r18_eight_variants.py", "-v"], cwd=ROOT, check=True)
    pending, active, completed = list(jobs), {}, []
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        while pending or active:
            free_gpus = [gpu for gpu in gpus[:max_parallel] if gpu not in active]
            while pending and free_gpus:
                job, gpu = pending.pop(0), free_gpus.pop(0)
                run_dir = Path(job["run_dir"])
                run_dir.mkdir(parents=True, exist_ok=True)
                log = (run_dir / "launcher_stdout.log").open("a", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                (run_dir / "launcher.pid").write_text(str(process.pid), encoding="utf-8")
                job.update({"gpu": gpu, "pid": process.pid,
                            "started_at": dt.datetime.now().astimezone().isoformat(), "status": "running"})
                active[gpu] = (job, process, log)
                print(f"started {job['model']} pid={process.pid} GPU={gpu}", flush=True)
            if not active:
                break
            time.sleep(2)
            for gpu, (job, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job.update({"return_code": code, "finished_at": dt.datetime.now().astimezone().isoformat(),
                            "status": "completed" if code == 0 else "failed"})
                completed.append(job)
                active.pop(gpu)
                Path(job["run_dir"], "launcher.pid").unlink(missing_ok=True)
                print(f"finished {job['model']} rc={code} GPU={gpu}", flush=True)
    except KeyboardInterrupt:
        for job, process, log in active.values():
            process.terminate()
            log.close()
            job.update({"status": "interrupted", "finished_at": dt.datetime.now().astimezone().isoformat()})
        raise
    manifest["jobs"] = completed + pending + [item[0] for item in active.values()]
    manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
    write_json(manifest_path, manifest)
    if any(job.get("status") == "failed" for job in completed):
        raise SystemExit("one or more R18 variant jobs failed; inspect launcher_stdout.log")


if __name__ == "__main__":
    main()
