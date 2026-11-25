#!/usr/bin/env bash
# Subprocess-Isolated Cross-Dataset Full-Metric Evaluator on RTX 4090
# Usage: bash RTDETR_CrossDataset_4090/run_eval_cross_4090.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

SEED=${SEED:-0}
DEVICE=${DEVICE:-0}
ROOT="runs/cross_dataset_4090"
OUT_JSONL="${ROOT}/cross_dataset_4090_results.jsonl"
rm -f "$OUT_JSONL"

for DS in TIB-Net Det-Fly NPS-Drones; do
  DATA_YAML="dataset/${DS}.yaml"
  if [ ! -f "$DATA_YAML" ]; then
    echo "[SKIP] Missing $DATA_YAML"
    continue
  fi
  for MDL in RT-DETR-R18 Exp2_PruneS4 Exp27_StarFast_NativeRepC3 Exp28_StarFast_HiLoHaar Exp14_StarHR_Turbo_D2; do
    CKPT="${ROOT}/${DS}/${MDL}_seed${SEED}/weights/best.pt"
    if [ -f "$CKPT" ]; then
      PYTHONPATH=. python RTDETR_CrossDataset_4090/eval_single_ckpt_4090.py \
        --dataset "$DS" \
        --data-yaml "$DATA_YAML" \
        --model-name "$MDL" \
        --ckpt "$CKPT" \
        --out-jsonl "$OUT_JSONL" \
        --split test \
        --imgsz 640 \
        --device "$DEVICE"
    fi
  done
done

python - "$OUT_JSONL" <<'EOF'
import csv, json, sys
from pathlib import Path

jsonl = Path(sys.argv[1])
if not jsonl.is_file():
    print("[WARN] No evaluation results found.")
    sys.exit(0)

rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
out_csv = jsonl.with_suffix(".csv")
out_json = jsonl.with_suffix(".json")
out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
if rows:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

print(f"\n==================== CROSS-DATASET 4090 SUMMARY ({len(rows)} RUNS) ====================")
print("| Dataset | Split | Model | Layers | Params(M) | GFLOPs | Pt(MB) | P | R | F1 | mAP50 | mAP75 | mAP50:95 | Val(ms) | FP16_b16(ms/FPS) | FP32_b1(ms/FPS) |")
print("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in rows:
    print(
        f"| {r['Dataset']} | {r['Split']} | {r['Model']} | {r['Layers']} | {r['Params_M']:.2f} | {r['GFLOPs']:.1f} | {r['Pt_MB']:.1f} | "
        f"{r['P']:.4f} | {r['R']:.4f} | {r['F1']:.4f} | {r['mAP50']:.4f} | {r['mAP75']:.4f} | {r['mAP50_95']:.4f} | "
        f"{r['Val_ms']:.2f} | {r['FP16_b16_ms']:.2f} / {r['FP16_b16_FPS']:.1f} | {r['FP32_b1_ms']:.2f} / {r['FP32_b1_FPS']:.1f} |"
    )
print(f"\nSaved: {out_csv} & {out_json}")
EOF
