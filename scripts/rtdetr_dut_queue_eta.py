"""Print approximate remaining time for the three-GPU, nine-model training queue.

Uses each run's results.csv (one row per completed epoch) and the launch
timestamp recorded in its log. No PyTorch dependency is needed.
"""

import argparse
import csv
import datetime as dt
import re
import time
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAVE_DIR = re.compile(r"\bsave_dir=([^,\s]+)")
START = re.compile(r"\bQUEUE_START_UNIX=(\d+)")
EPOCHS = re.compile(r"(?m)^epochs:\s*(\d+)\s*$")


def elapsed_text(seconds):
    minutes = max(0, round(seconds / 60))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    parts.append(f"{mins}分钟")
    return "".join(parts)


def run_state(log_file, now):
    if not log_file.exists():
        return None
    with log_file.open("r", encoding="utf-8", errors="replace") as stream:
        prefix = ANSI.sub("", stream.read(512 * 1024))
    start_match = START.search(prefix)
    save_matches = SAVE_DIR.findall(prefix)
    if not start_match:
        return None
    start = int(start_match.group(1))
    if not save_matches:
        return {"epochs": None, "completed": 0, "remaining": None, "rate": None}

    run_dir = Path(save_matches[-1])
    epochs = 150
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        match = EPOCHS.search(args_file.read_text(encoding="utf-8", errors="replace"))
        if match:
            epochs = int(match.group(1))

    completed = 0
    csv_file = run_dir / "results.csv"
    if csv_file.exists():
        with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            for row in csv.DictReader(stream):
                value = next((v for k, v in row.items() if k and k.strip() == "epoch"), None)
                if value:
                    try:
                        completed = max(completed, int(float(value)))
                    except ValueError:
                        pass

    if completed == 0:
        return {"epochs": epochs, "completed": 0, "remaining": None, "rate": None}
    rate = max(1.0, (now - start) / completed)
    remaining = max(0, epochs - completed) * rate
    return {"epochs": epochs, "completed": completed, "remaining": remaining, "rate": rate}


def print_eta(log_dir, wave, configs, gpus):
    now = time.time()
    start_index = (wave - 1) * 3
    states = []
    print(f"\n[ETA {dt.datetime.now():%F %T}] 第 {wave}/3 轮（基于已完成 epoch 的粗略估算）", flush=True)
    for slot in range(3):
        config = configs[start_index + slot]
        gpu = gpus[slot]
        label = f"{Path(config).stem}@GPU{gpu}"
        state = run_state(log_dir / f"{Path(config).stem}_gpu{gpu}.log", now)
        states.append(state)
        if state is None or state["epochs"] is None:
            detail = "初始化中，等待训练记录"
        elif state["remaining"] is None:
            detail = f"0/{state['epochs']} epoch，等待首个 epoch 完成"
        elif state["completed"] >= state["epochs"]:
            detail = f"{state['completed']}/{state['epochs']} epoch，收尾验证中"
        else:
            detail = (f"{state['completed']}/{state['epochs']} epoch，"
                      f"预计剩余 {elapsed_text(state['remaining'])}")
        print(f"  {label}: {detail}", flush=True)

    if all(state and state["remaining"] is not None for state in states):
        current_wave = max(state["remaining"] for state in states)
        slowest_epoch = max(state["rate"] for state in states)
        full_wave = slowest_epoch * max(state["epochs"] for state in states)
        queue_remaining = current_wave + (3 - wave) * full_wave
        finish = dt.datetime.now() + dt.timedelta(seconds=queue_remaining)
        print(f"  本轮预计剩余 {elapsed_text(current_wave)}；"
              f"全部9项预计剩余 {elapsed_text(queue_remaining)}，"
              f"约 {finish:%F %T} 完成", flush=True)
        if wave < 3:
            print("  未启动模型按本轮最慢速度估算，实际耗时可能明显不同。", flush=True)
    else:
        print("  全队列 ETA 将在本轮三个模型都完成首个 epoch 后显示。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--wave", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--configs", nargs=9, required=True)
    parser.add_argument("--gpus", nargs=3, required=True)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    while True:
        try:
            print_eta(args.log_dir, args.wave, args.configs, args.gpus)
        except (OSError, csv.Error) as exc:
            print(f"[ETA] 暂时无法读取训练进度：{exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
