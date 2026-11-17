"""Run: python -m pytest -q tests/test_signals.py (torch installed)."""
import importlib.util
from pathlib import Path
import torch

p = Path(__file__).resolve().parents[1] / 'src' / 'dut_sparse_signals.py'
spec = importlib.util.spec_from_file_location('dut_sparse_signals', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DUTSparseSignals = mod.DUTSparseSignals


def data(with_gt=True):
    torch.manual_seed(9)
    b,q,c,l = 2,12,1,4
    box=torch.rand(l,b,q,4)*0.75+0.10
    box[...,2:]*=.18
    box.requires_grad_()
    scores=torch.randn(l,b,q,c,requires_grad=True)
    dense=torch.randn(b,4*4+2*2+1,c,requires_grad=True)
    if with_gt:
        gt=torch.tensor([[0.49,0.53,.017,.022],[0.3,0.2,.09,.08]])
        batch={'gt_groups':[1,1],'bboxes':gt,'cls':torch.tensor([0,0])}
        matches=[(torch.tensor([1]),torch.tensor([0])),(torch.tensor([3]),torch.tensor([1]))]
    else:
        batch={'gt_groups':[0,0],'bboxes':torch.empty(0,4),'cls':torch.empty(0,dtype=torch.long)}
        matches=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(b)]
    return box,scores,dense,batch,matches


def test_zero_defaults():
    x,s,d,b,m=data()
    loss=DUTSparseSignals()(x,s,b,m,d,[(4,4),(2,2),(1,1)])
    assert set(loss)=={'loss_gap','loss_hnr','loss_plm','loss_bqr'}
    assert all(float(v)==0 for v in loss.values())


def test_all_terms_finite_grad():
    x,s,d,b,m=data()
    crit=DUTSparseSignals(gaussian_weight=.1,rank_weight=.05,
                          monotonic_weight=.03,jitter_weight=.05)
    terms=crit(x,s,b,m,d,[(4,4),(2,2),(1,1)])
    assert all(torch.isfinite(v) for v in terms.values())
    loss=sum(terms.values())
    loss.backward()
    for arr in (x,s,d):
        assert arr.grad is not None and torch.isfinite(arr.grad).all()
    assert d.grad.abs().sum()>0
    assert x.grad.abs().sum()>0
    assert s.grad.abs().sum()>0


def test_no_gt():
    x,s,d,b,m=data(False)
    terms=DUTSparseSignals(gaussian_weight=.1,rank_weight=.05,
                           monotonic_weight=.03,jitter_weight=.05)(
                           x,s,b,m,d,[(4,4),(2,2),(1,1)])
    assert all(torch.isfinite(v) for v in terms.values())
    assert sum(terms.values()).backward() is None


def test_pairwise_iou():
    x=torch.tensor([[.5,.5,.1,.1]])
    assert abs(mod.pairwise_iou(x,x)[0,0].item()-1)<1e-5
    assert abs(mod.aligned_giou(x,x)[0].item()-1)<1e-5
    y=torch.tensor([[.1,.1,.1,.1]])
    assert mod.pairwise_iou(x,y)[0,0] == 0


def test_shape_guard():
    x,s,d,b,m=data()
    try:
        DUTSparseSignals(gaussian_weight=.1)(x,s,b,m,d,[(2,2)])
    except ValueError:
        pass
    else:
        assert False, 'expected dense spatial dimension guard'
