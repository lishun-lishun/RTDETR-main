# RTDETR Cross-Dataset Suite (`Det-Fly`, `NPS-Drones`, `TIB-Net`)

## File List
- `prepare_cross_datasets.py`: Auto-inspects raw folders, prevents video-frame leakage via group-aware splitting (`seed=0`), creates symlinked YOLO datasets without touching raw files, and writes `dataset/{Det-Fly,NPS-Drones,TIB-Net}.yaml`.
- `train_cross_dataset.py`: Unified fair trainer (`RT-DETR-R18` vs `Exp12_StarHR_Balanced`, `epochs=150, batch=8, imgsz=640, AdamW, lr0=1e-4, cos_lr=True, mosaic=0.0, seed=0`).
- `eval_cross_dataset.py`: Full-metric evaluator (`P, R, mAP50, mAP75, mAP50:95, Layers, Params, GFLOPs, Pt_MB, Val_ms, FP16_b16, FP16_b1, FP32_b1`) saving Markdown, CSV, and JSON.
- `run_6gpu_cross_dataset.sh`: End-to-end 6-GPU parallel launcher (`GPUs 0-5`) followed by unified evaluation.

## Quick Start
```bash
cd /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main
python RTDETR_CrossDataset_Suite/prepare_cross_datasets.py --repo .
bash RTDETR_CrossDataset_Suite/run_6gpu_cross_dataset.sh
```
