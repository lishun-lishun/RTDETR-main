from __future__ import annotations
import unittest
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/'ultralytics/cfg/models/rt-detr'
NAMES={'r17a_cbr':'rtdetr-r17a-cbr.yaml','r17b_hrd_512':'rtdetr-r17b-hrd-512.yaml','r17c_hrd_cbr':'rtdetr-r17c-hrd-cbr.yaml'}
class TestR17HRD(unittest.TestCase):
 def test_yaml_inventory_and_original_unchanged(self):
  base=yaml.safe_load((MODEL_DIR/'rtdetr-r18.yaml').read_text(encoding='utf-8'))
  self.assertEqual(base['head'][-1][0],[19,22,25])
  for name,file in NAMES.items():
   d=yaml.safe_load((MODEL_DIR/file).read_text(encoding='utf-8')); self.assertTrue((MODEL_DIR/file).is_file())
   self.assertEqual(d['nc'],1)
   self.assertEqual(d['head'][-1][2],'RTDETRDecoder')
   self.assertEqual(d['head'][-1][1],1)
   self.assertEqual(d['head'][-1][3][1:],[256,300,4,8,3])
 def test_stage5_channel_controls(self):
  a=yaml.safe_load((MODEL_DIR/NAMES['r17a_cbr']).read_text(encoding='utf-8'))
  b=yaml.safe_load((MODEL_DIR/NAMES['r17b_hrd_512']).read_text(encoding='utf-8'))
  c=yaml.safe_load((MODEL_DIR/NAMES['r17c_hrd_cbr']).read_text(encoding='utf-8'))
  self.assertEqual(a['backbone'][7][3][0],256); self.assertEqual(b['backbone'][12][3][0],512); self.assertEqual(c['backbone'][12][3][0],256)
  self.assertEqual([x[2] for x in b['backbone'][6:12]],['HRStreamInit','HRUpdate3','HRInjectP3','Blocks','HRUpdate4','HRInjectP4'])
  self.assertEqual(b['backbone'][7][0],[6,5]); self.assertEqual(b['backbone'][8][0],[5,7])
  self.assertEqual(b['backbone'][10][0],[7,9]); self.assertEqual(b['backbone'][11][0],[9,10])
 def test_module_zero_identity_and_amp(self):
  try: import torch
  except ImportError: self.skipTest('PyTorch is not installed')
  from ultralytics.nn.modules import HRInjectP3,HRInjectP4
  s3=torch.randn(2,128,80,80); h=torch.randn(2,64,80,80); m=HRInjectP3(128,64).eval(); self.assertEqual(float((m([s3,h])-s3).abs().max()),0.0)
  s4=torch.randn(2,256,40,40); m4=HRInjectP4(256,64).eval(); self.assertEqual(float((m4([s4,h])-s4).abs().max()),0.0)
 def test_full_model_contracts(self):
  try: import torch; from ultralytics.nn.tasks import RTDETRDetectionModel
  except ImportError: self.skipTest('PyTorch is not installed')
  for name,file in [('baseline','rtdetr-r18.yaml'),*NAMES.items()]:
   model=RTDETRDetectionModel(str(MODEL_DIR/file),nc=1,verbose=False).eval(); found={}; indices=(5,6,7) if name in ('baseline','r17a_cbr') else (8,11,12)
   hs=[model.model[i].register_forward_hook(lambda _m,_i,o,k=k:found.__setitem__(k,tuple(o.shape))) for i,k in zip(indices,('p3','p4','p5'))]
   try:
    with torch.inference_mode(): model(torch.zeros(1,3,640,640))
   finally:
    for h in hs:h.remove()
   self.assertEqual([found[k] for k in ('p3','p4','p5')],[(1,128,80,80),(1,256,40,40),(1,512 if name in ('baseline','r17b_hrd_512') else 256,20,20)])
if __name__=='__main__': unittest.main()
