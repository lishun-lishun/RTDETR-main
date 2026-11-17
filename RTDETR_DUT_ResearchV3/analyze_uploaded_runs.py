"""Recompute fair epoch-matched RT-DETR ablation summaries from local CSVs.

Usage:
  python analyze_uploaded_runs.py --repo /path/to/RTDETR-main --out reports/v3_matched_epoch.csv
Read-only on runs/; output written to an explicit new report path.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

RUNS=['e0_baseline','e1_queryonly','e2_evidence_refine','e3_quality','e4_loss_only',
      'e5_all','e6_p2_queryonly','e7_p2_evidence','e8_p2_all',
      'tinyv2_m1_peak','tinyv2_m2_dn','tinyv2_m3_match','tinyv2_m4_nwd']
KEY='metrics/mAP50-95(B)'

def load_one(path):
    if not path.is_file():return None
    with path.open(newline='',encoding='utf8') as fp:
        reader=csv.DictReader(fp)
        records=[{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    out={}
    for row in records:
        if not row.get(KEY):continue
        e=int(float(row['epoch']))
        out[e]=float(row[KEY])
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--out',default='reports/v3_matched_epoch.csv',type=Path)
    opts=ap.parse_args()
    repo=opts.repo.resolve()
    loaded={name:load_one(repo/'runs/uav_ablation'/name/'seed0/results.csv') for name in RUNS}
    base=loaded['e0_baseline']
    if not base:ap.error('E0 baseline not found at provided repo')
    epochs=(40,68,70,100,120,123,150)
    output=[]
    for name,vals in loaded.items():
        if not vals:
            output.append(dict(experiment=name,status='missing'))
            continue
        best_ep=max(vals,key=vals.get)
        row=dict(experiment=name,status='ok',last_epoch=max(vals),best_epoch=best_ep,best_val_ap5095=vals[best_ep])
        for epoch in epochs:
            row[f'epoch_{epoch}']=vals.get(epoch,'')
            row[f'delta_vs_E0_epoch_{epoch}']=round(vals[epoch]-base[epoch],6) if epoch in base and epoch in vals else ''
        output.append(row)
    dest=opts.out if opts.out.is_absolute() else repo/opts.out
    if dest.exists():ap.error('Refusing to overwrite existing report: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in output for k in row))
    with dest.open('w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(output)
    print('Wrote',dest)
    print('WARNING: E0 has only',max(base),'logged epochs; no unbiased 150-epoch E0 comparison until it finishes.')
    for row in output:
        print(f"{row['experiment']:<23} {row.get('last_epoch','—')} epochs best {row.get('best_val_ap5095','—')} ",
              'same120:',row.get('delta_vs_E0_epoch_120','—'))

if __name__=='__main__':main()
