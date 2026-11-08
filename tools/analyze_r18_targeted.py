"""Summarize targeted validation metrics and diagnostics into CSV/Markdown."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from r18_targeted_common import CONFIGS, RUNS, config_path, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',type=Path,default=RUNS); p.add_argument('--output',type=Path,default=RUNS/'analysis'); p.add_argument('--models',default='all'); a=p.parse_args()
    names=list(CONFIGS) if a.models=='all' else [x.strip() for x in a.models.split(',') if x.strip()]
    rows=[]
    for n in names:
        metric=a.runs/n/'seed0'/'eval'/'evaluation_metrics.json'
        loc=a.runs/n/'seed0'/'eval'/'localization_analysis.json'
        rec={'model':n,'yaml':str(config_path(n)),'description':CONFIGS[n],'metrics_file':str(metric),'exists':metric.is_file(),
             'localization_file':str(loc),'localization_exists':loc.is_file()}
        if metric.is_file():
            try: rec.update({k:v for k,v in json.loads(metric.read_text(encoding='utf-8')).items() if k in ('mAP50','mAP50:95','precision','recall','split')})
            except Exception as exc: rec['error']=str(exc)
        if loc.is_file():
            try:
                lv=json.loads(loc.read_text(encoding='utf-8'))
                rec.update({k:lv.get(k) for k in ('mean_best_iou','mean_center_error','mean_width_error','mean_height_error','small_object_mean_iou')})
            except Exception as exc: rec['localization_error']=str(exc)
        rows.append(rec)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'targeted_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    write_json(a.output/'targeted_summary.json',rows); print(json.dumps(rows,indent=2,default=str))

if __name__=='__main__': main()
