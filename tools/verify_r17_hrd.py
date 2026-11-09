"""Verify HRD-R18 R17 YAML contracts, zero-init identity and gradients.

This script is explicit and read-only by default.  It never starts training.
"""
from __future__ import annotations
import argparse, json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'ultralytics/cfg/models/rt-detr'
CONFIGS={'baseline':'rtdetr-r18.yaml','r17a_cbr':'rtdetr-r17a-cbr.yaml','r17b_hrd_512':'rtdetr-r17b-hrd-512.yaml','r17c_hrd_cbr':'rtdetr-r17c-hrd-cbr.yaml'}
EXPECTED={'baseline':{'layers':(5,6,7),'decoder':[19,22,25]},'r17a_cbr':{'layers':(5,6,7),'decoder':[19,22,25]},'r17b_hrd_512':{'layers':(8,11,12),'decoder':[24,27,30]},'r17c_hrd_cbr':{'layers':(8,11,12),'decoder':[24,27,30]}}

def local_device(value):
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
 ids=[x.strip() for x in visible.split(',') if x.strip()]
 if value.isdigit() and ids:
  if len(ids)==1: return '0'
  if value in ids: return str(ids.index(value))
 return value

def shape_check(name, device, imgsz):
 import torch
 from ultralytics.nn.tasks import RTDETRDetectionModel
 model=RTDETRDetectionModel(str(MODEL_DIR/CONFIGS[name]),nc=1,verbose=False).to(device).eval()
 found={}; dec={}
 handles=[]
 for idx,key in zip(EXPECTED[name]['layers'],('p3','p4','p5')):
  handles.append(model.model[idx].register_forward_hook(lambda _m,_i,o,key=key: found.__setitem__(key,tuple(o.shape))))
 handles.append(model.model[-1].register_forward_pre_hook(lambda _m,v: dec.__setitem__('inputs',[tuple(x.shape) for x in v[0]])))
 try:
  with torch.inference_mode(): model(torch.randn(2,3,imgsz,imgsz,device=device))
 finally:
  for h in handles: h.remove()
 result={'backbone':[found[k] for k in ('p3','p4','p5')],'decoder':dec.get('inputs'),'decoder_from':list(model.model[-1].f)}
 expected={'backbone':[(2,128,imgsz//8,imgsz//8),(2,256,imgsz//16,imgsz//16),(2,512 if name in ('baseline','r17b_hrd_512') else 256,imgsz//32,imgsz//32)],'decoder':[(2,256,imgsz//8,imgsz//8),(2,256,imgsz//16,imgsz//16),(2,256,imgsz//32,imgsz//32)],'decoder_from':EXPECTED[name]['decoder']}
 if result != expected: raise AssertionError(f'{name}: {result} != {expected}')
 return result

def zero_identity_and_grads(device):
 import torch
 from ultralytics.nn.modules import HRInjectP3, HRInjectP4, HRStreamInit, HRUpdate3, HRUpdate4
 checks={}
 s3=torch.randn(2,128,80,80,device=device); h=torch.randn(2,64,80,80,device=device)
 p3=HRInjectP3(128,64).to(device).eval(); checks['p3_zero_diff']=float((p3([s3,h])-s3).abs().max())
 s4=torch.randn(2,256,40,40,device=device); p4=HRInjectP4(256,64).to(device).eval(); checks['p4_zero_diff']=float((p4([s4,h])-s4).abs().max())
 modules=[HRStreamInit(64,64),HRUpdate3(128,64),HRUpdate4(256,64),HRInjectP3(128,64),HRInjectP4(256,64)]
 for m in modules: m.to(device).train()
 loss=modules[0](torch.randn(2,64,160,160,device=device)).square().mean()
 loss=loss+modules[1]([s3,h]).square().mean()+modules[2]([s4,h]).square().mean()
 loss=loss+modules[3]([s3,h]).square().mean()+modules[4]([s4,h]).square().mean(); loss.backward()
 checks['trainable_nonzero_grads']=all(p.grad is not None and torch.isfinite(p.grad).all() for m in modules for p in m.parameters() if p.requires_grad)
 checks['terminal_projection_bn_grads']=all(m.zero_proj.norm.weight.grad is not None and m.zero_proj.norm.weight.grad.abs().sum()>0 for m in (modules[3],modules[4]))
 return checks

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--models',default='all'); p.add_argument('--device',default='0'); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--output',type=Path); a=p.parse_args(); a.device=local_device(a.device)
 names=list(CONFIGS) if a.models=='all' else [x.strip() for x in a.models.split(',') if x.strip()]
 unknown=[x for x in names if x not in CONFIGS]
 if unknown:p.error(f'unknown models {unknown}')
 missing=[x for x in names if not (MODEL_DIR/CONFIGS[x]).is_file()]
 if missing:p.error(f'missing configs {missing}')
 try:
  import torch
  device=torch.device(f'cuda:{a.device}' if str(a.device).isdigit() else a.device)
  if device.type=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
  results={'shapes':{n:shape_check(n,device,a.imgsz) for n in names},'zero_identity_and_gradients':zero_identity_and_grads(device)}
 except ImportError as exc:
  p.error(f'PyTorch is required for runtime verification: {exc}')
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
 print(json.dumps(results,indent=2,default=str))

if __name__=='__main__': main()
