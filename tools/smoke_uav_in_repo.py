"""Run after installing the patch inside the ACTUAL RTDETR-main Python env.

Examples:
  python tools/smoke_in_repo.py --yaml ultralytics/cfg/models/rt-detr/rtdetr_r18_uav_evidence_refine.yaml
  python tools/smoke_in_repo.py --yaml ultralytics/cfg/models/rt-detr/rtdetr_r18_uav_loss_only.yaml --backward
"""
from __future__ import annotations
import argparse
import torch
from ultralytics import RTDETR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--backward', action='store_true')
    args = ap.parse_args()
    m = RTDETR(args.yaml).model.to(args.device)
    x = torch.rand(1, 3, 640, 640, device=args.device)
    m.eval()
    with torch.no_grad():
        y = m.predict(x)
        if isinstance(y, tuple): y = y[0]
        assert y.shape == (1, 300, 4 + m.nc), y.shape
        assert torch.isfinite(y).all(), 'NaN or Inf in predictions'
        print('EVAL PASS',tuple(y.shape))
    if args.backward:
        m.train()
        # RTDETRDetectionModel.loss expects ground truth normalized xywh.
        batch = {
            'img': x,
            'cls': torch.tensor([[0]],dtype=torch.float32,device=args.device),
            'bboxes': torch.tensor([[.5,.5,.04,.04]],dtype=torch.float32,device=args.device),
            'batch_idx': torch.tensor([0],dtype=torch.long,device=args.device),
        }
        loss, terms = m.loss(batch)
        assert torch.isfinite(loss).item()
        loss.backward()
        custom = [p for n,p in m.named_parameters() if 'evidence.' in n or 'refiner.' in n or 'reranker.' in n]
        print('TRAIN PASS',float(loss.detach()),'custom params with grad',sum(p.grad is not None for p in custom))

if __name__ == '__main__':
    main()
