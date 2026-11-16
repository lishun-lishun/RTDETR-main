"""Run key functions with minimal API stubs because target repo is not in runtime."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]

def attach(name, **values):
    mod=types.ModuleType(name)
    for key,value in values.items():
        setattr(mod,key,value)
    sys.modules[name]=mod
    return mod

class MockMatcher(torch.nn.Module):
    def __init__(self,cost_gain=None,use_fl=True,with_mask=False,num_sample_points=12544,alpha=.25,gamma=2.):
        super().__init__()
        self.cost_gain=cost_gain or {'class':2,'bbox':5,'giou':2}
        self.use_fl=use_fl; self.alpha=alpha; self.gamma=gamma; self.with_mask=with_mask
    def forward(self,*args,**kwargs):
        raise NotImplementedError

class MockLoss:
    def __init__(self,nc=1,**kwargs):
        self.matcher=MockMatcher()
        self.nwd_loss=False; self.gcd_loss=False
    def _get_loss_bbox(self,p,g,postfix=''):
        return {f'loss_giou{postfix}':(p-g).square().mean(),f'loss_bbox{postfix}':(p-g).abs().mean()}


def iou_xywh(a,b,xywh=True,GIoU=False):
    ax1=a[...,:2]-a[...,2:]/2; ax2=a[...,:2]+a[...,2:]/2
    bx1=b[...,:2]-b[...,2:]/2; bx2=b[...,:2]+b[...,2:]/2
    inter=(torch.minimum(ax2,bx2)-torch.maximum(ax1,bx1)).clamp(min=0).prod(-1)
    area_a=a[...,2:].prod(-1); area_b=b[...,2:].prod(-1)
    return inter/(area_a+area_b-inter+1e-7)


def load_loss():
    for m in ['ultralytics','ultralytics.models','ultralytics.models.utils','ultralytics.utils']:
        sys.modules.setdefault(m,types.ModuleType(m))
    attach('ultralytics.models.utils.loss',RTDETRDetectionLoss=MockLoss)
    attach('ultralytics.models.utils.ops',HungarianMatcher=MockMatcher)
    attach('ultralytics.utils.metrics',bbox_iou=iou_xywh)
    spec=importlib.util.spec_from_file_location('isolated_tiny_loss',ROOT/'src/tiny_uav_loss.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def test_center_match_two_images():
    mod=load_loss()
    matcher=mod.TinyCenterHungarianMatcher(center_weight=.6,cost_gain={'class':2,'bbox':5,'giou':2})
    pred=torch.tensor([[[.5,.5,.02,.02],[.2,.2,.02,.02],[.9,.9,.02,.02]],
                       [[.2,.2,.02,.02],[.8,.8,.02,.02],[.1,.1,.02,.02]]])
    logit=torch.zeros(2,3,1)
    gt=torch.tensor([[.5,.5,.02,.02],[.2,.2,.02,.02],[.8,.8,.02,.02]])
    result=matcher(pred,logit,gt,torch.zeros(3,dtype=torch.long),[1,2])
    assert result[0][0].tolist()==[0]
    assert result[0][1].tolist()==[0]
    assert result[1][1].tolist()==[1,2]


def test_loss_nwd_gradient_and_zero_gt():
    mod=load_loss()
    loss=mod.TinyUAVRTDETRLoss(nc=1,settings={'nwd_weight':1})
    p=torch.tensor([[.51,.5,.02,.02]],requires_grad=True)
    g=torch.tensor([[.5,.5,.02,.02]])
    result=loss._get_loss_bbox(p,g)
    value=sum(result.values());value.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    empty=loss._get_loss_bbox(torch.zeros(0,4),torch.zeros(0,4))
    assert 'loss_giou' in empty


def test_peak_rank_multiscale_and_pixel_dn():
    for m in ['ultralytics.nn','ultralytics.nn.modules']:
        sys.modules.setdefault(m,types.ModuleType(m))
    attach('ultralytics.nn.modules.head',RTDETRDecoder=object)
    attach('ultralytics.models.utils.ops',get_cdn_group=lambda *args:None)
    def xywh2xyxy(b):
        return torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
    def xyxy2xywh(b):
        return torch.cat(((b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]),-1)
    attach('ultralytics.utils.ops',xywh2xyxy=xywh2xyxy,xyxy2xywh=xyxy2xywh)
    spec=importlib.util.spec_from_file_location('isolated_tiny_decoder',ROOT/'src/tiny_uav_decoder.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    dec=object.__new__(obj.TinyUAVDecoder)
    dec.peak_alpha=.6;dec.peak_kernel=3
    raw=torch.randn(2,16+4+1)
    out=obj.TinyUAVDecoder._peak_rank(dec,raw,[[4,4],[2,2],[1,1]])
    assert out.shape==raw.shape and torch.isfinite(out).all()
    dec.dn_pos_px=1.25;dec.dn_neg_px=4.;dec.input_pixels=640
    batch={'gt_groups':[1,2],'batch_idx':torch.tensor([0,1,1]),
           'bboxes':torch.tensor([[.5,.5,.02,.02],[.2,.2,.01,.02],[.8,.8,.02,.03]])}
    dn=torch.zeros(2,8,4)
    # 2 groups, maximum 2 per batch -> 8 total DN slots.
    meta={'dn_num_group':2, 'dn_pos_idx':[torch.tensor([0,2]),torch.tensor([0,1,2,3])]}
    new=obj.TinyUAVDecoder._pixel_dn(dec,dn,batch,meta)
    assert new.shape==dn.shape and torch.isfinite(new).all()
    assert torch.count_nonzero(new).item()>0
    # Non-GT pad remains identical
    assert torch.equal(new[0,1],dn[0,1])
