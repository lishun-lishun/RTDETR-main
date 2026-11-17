"""Run opt-in DUT experiments; DRY RUN by default. Execute from repository root.

Example: python /path/to/pack/run_v3.py --repo . --exp gap --device 0 --execute
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

NAME_TO_CONFIG={n:f'rtdetr_r18_dut_v3_{n}.yaml' for n in
                ['gap','hnr','plm','bqr','gap_hnr','all']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--exp',choices=NAME_TO_CONFIG,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--run-name',default=None,help='distinct output name for retries; seed remains unchanged')
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    root=a.repo.resolve()
    model=root/'ultralytics/cfg/models/rt-detr'/NAME_TO_CONFIG[a.exp]
    data=root/'dataset/DUT-Anti-UAV.yaml'
    dest=root/'runs/dut_sparse_v3'/a.exp/(a.run_name or f'seed{a.seed}')
    if not model.is_file() or not data.is_file(): p.error('Model YAML or data YAML missing')
    opts=dict(data=str(data),epochs=a.epochs,patience=0,batch=8,imgsz=640,cache=False,
              workers=4,pretrained=False,optimizer='AdamW',seed=a.seed,
              deterministic=True,cos_lr=False,close_mosaic=0,amp=False,
              lr0=.0001,lrf=1.,momentum=.9,weight_decay=.0001,
              warmup_epochs=2000,warmup_momentum=.8,warmup_bias_lr=0.,
              max_det=300,mosaic=0.,mixup=0.,copy_paste=0.,device=a.device,
              project=str(dest.parent),name=dest.name,exist_ok=False)
    print(json.dumps({'model':str(model),'destination':str(dest),'options':opts},indent=2))
    if not a.execute:
        print('DRY RUN. Add --execute to train.')
        return
    if dest.exists(): p.error('Output already exists; do not overwrite runs')
    import os, sys
    os.chdir(root)
    sys.path.insert(0,str(root))
    import numpy as np, torch
    from ultralytics import RTDETR
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    model_obj=RTDETR(str(model))
    model_obj.train(**opts)

if __name__=='__main__':main()
