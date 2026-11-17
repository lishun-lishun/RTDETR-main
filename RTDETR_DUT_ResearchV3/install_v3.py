"""Install a *minimal*, opt-in training-only research extension into RTDETR-main.

DRY RUN by default; --apply creates new files and patches only one method in tasks.py.
No original YAML/checkpoint/runs is overwritten. No remote network interaction.
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
EXPS={
    'gap': {'gaussian_weight': .05},
    'hnr': {'rank_weight': .02},
    'plm': {'monotonic_weight': .025},
    'bqr': {'jitter_weight': .05},
    'gap_hnr': {'gaussian_weight': .05, 'rank_weight': .02},
    'all': {'gaussian_weight': .05, 'rank_weight': .02,
            'monotonic_weight': .025, 'jitter_weight': .05},
}
DEFAULTS={'gaussian_weight':0.,'rank_weight':0.,'monotonic_weight':0.,
          'jitter_weight':0., 'tiny_threshold_px':24., 'hard_neg_topk':8,
          'margin':.25, 'monotonic_margin':.025,'jitter_px':.5,
          'min_sigma_cells':.8,'negative_iou':.25}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--apply',action='store_true',help='write files (default only preview)')
    args=p.parse_args()
    repo=args.repo.expanduser().resolve()
    tasks=repo/'ultralytics/nn/tasks.py'
    basemodel=repo/'ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml'
    target=repo/'ultralytics/models/utils'
    if not tasks.is_file() or not basemodel.is_file() or not target.is_dir():
        p.error('Expected RTDETR-main structure: tasks.py / original R18 YAML / models/utils')
    old=tasks.read_text(encoding='utf-8')
    anchor=("return RTDETRDetectionLoss(nc=self.nc, use_vfl=True, use_sl=False, "
            "use_emasl=False, use_svfl=False, use_emasvfl=False, use_mal=False)")
    marker='DUT-SPARSE-V3 OPT-IN'
    original_forward='preds = self.predict(img, batch=targets) if preds is None else preds'
    original_loss_call='''loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)'''
    if marker not in old and (old.count(anchor)!=1 or old.count(original_forward)!=1
                              or old.count(original_loss_call)!=1):
        p.error('Cannot locate unique criterion/forward anchors; stop and inspect')
    if marker in old:
        print('Criterion hook already installed; will not patch tasks.py again.')
        new=old
    else:
        # The extra criterion is only constructed for opt-in YAMLs.
        insertion='''# DUT-SPARSE-V3 OPT-IN -- training only; original R18 logic unchanged below.
        if self.yaml.get('dut_sparse_v3'):
            from ultralytics.models.utils.dut_sparse_supervision import DUTSparseCriterion
            criterion = DUTSparseCriterion(
                nc=self.nc, use_vfl=True, use_sl=False, use_emasl=False,
                use_svfl=False, use_emasvfl=False, use_mal=False,
                settings=self.yaml['dut_sparse_v3'])
            return criterion
        '''+anchor
        new=old.replace(anchor,insertion,1)
        forward_patch="""# DUT-SPARSE-V3 opt-in temporary encoder hook (no persistent hooks).
        if self.yaml.get('dut_sparse_v3') and preds is None:
            with self.criterion.capture_head(self.model[-1]):
                preds = self.predict(img, batch=targets)
        else:
            preds = self.predict(img, batch=targets) if preds is None else preds"""
        new=new.replace(original_forward,forward_patch,1)
        new=new.replace(original_loss_call, '''extra_kwargs = {'apply_extra': self.training} if self.yaml.get('dut_sparse_v3') else {}
        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta,
                              **extra_kwargs)''', 1)
    outputs={target/'dut_sparse_signals.py':HERE/'src/dut_sparse_signals.py',
             target/'dut_sparse_supervision.py':HERE/'src/dut_sparse_supervision.py'}
    yaml_outputs={}
    base=yaml.safe_load(basemodel.read_text(encoding='utf-8'))
    for name,changes in EXPS.items():
        cfg=yaml.safe_load(basemodel.read_text(encoding='utf-8'))
        cfg['dut_sparse_v3']={**DEFAULTS,**changes}
        yaml_outputs[basemodel.parent/f'rtdetr_r18_dut_v3_{name}.yaml']=yaml.safe_dump(
            cfg,sort_keys=False,allow_unicode=True)
    to_create=list(outputs)+list(yaml_outputs)
    collisions=[str(path) for path in to_create if path.exists()]
    print('Repository:',repo)
    print('tasks.py patch:', 'needed' if old!=new else 'already installed')
    print('Files to create:',len(to_create))
    for path in to_create: print(' +',path.relative_to(repo))
    if collisions:
        print('Already exist (will not overwrite):',collisions)
        if len(collisions)!=len(to_create):
            p.error('Partial installation found; inspect before retrying')
    if not args.apply:
        print('DRY RUN; no changes written. Use --apply when ready.')
        return
    if collisions and old==new:
        print('Already installed. No changes made.')
        return
    if collisions:
        p.error('Refusing to overwrite existing files')
    # Safety: original model YAML and existing checkpoints are never modified.
    for dst,src in outputs.items():
        shutil.copyfile(src,dst)
    for dst,s in yaml_outputs.items():
        dst.write_text(s,encoding='utf-8')
    if old!=new:
        backup=tasks.with_suffix('.py.dut_v3_backup')
        if backup.exists():
            p.error('Previous backup exists, refusing to overwrite')
        backup.write_text(old,encoding='utf-8')
        tasks.write_text(new,encoding='utf-8')
        print('Backup:',backup)
    print('Installed. Run original R18 smoke test and dedicated integration test first.')

if __name__=='__main__':main()
