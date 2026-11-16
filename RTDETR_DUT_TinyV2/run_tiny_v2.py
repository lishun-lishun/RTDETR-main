"""Small launcher; execute inside the target RTDETR-main repository root."""
import argparse
from pathlib import Path
from ultralytics import RTDETR


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--yaml', required=True)
    p.add_argument('--data', default='dataset/DUT-Anti-UAV.yaml')
    p.add_argument('--weights', default=None)
    p.add_argument('--device', default='0')
    p.add_argument('--name', required=True)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr0', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=float, default=3.)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--execute', action='store_true')
    args=p.parse_args()
    yaml=Path(args.yaml)
    if not yaml.exists():
        raise FileNotFoundError(yaml)
    print('CONFIG:',vars(args))
    if not args.execute:
        print('DRY RUN: pass --execute after baseline protocol audit')
        return
    model=RTDETR(str(yaml))
    if args.weights:
        model.load(args.weights)
    model.train(data=args.data,epochs=args.epochs,batch=args.batch,
                imgsz=640,device=args.device,seed=args.seed,
                lr0=args.lr0,warmup_epochs=args.warmup_epochs,
                amp=args.amp,project='runs/tinyv2',name=args.name,
                exist_ok=False)

if __name__=='__main__':
    main()
