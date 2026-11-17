"""Plug-in criterion for lishun-lishun/RTDETR-main (Ultralytics RT-DETR fork).

No modifications to RTDETRDecoder, original matcher, backbone, or inference.
Each extra loss is opt-in through YAML top-level 'dut_sparse_v3'.
"""
from __future__ import annotations
from contextlib import contextmanager
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from .dut_sparse_signals import DUTSparseSignals


class DUTSparseCriterion(RTDETRDetectionLoss):
    def __init__(self, *args, settings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals = DUTSparseSignals(**(settings or {}))
        self._dense_logits = None
        self._shapes = []
        self.last_extra = {}

    @contextmanager
    def capture_head(self, head):
        """Capture dense encoder scores only during model prediction.

        Hooks are removed BEFORE checkpointing, preventing unpicklable
        local-closure hooks from entering EMA or checkpoint state.
        """
        if not self.signals.gaussian_weight:
            yield
            return
        if not hasattr(head, 'enc_score_head') or not hasattr(head, 'input_proj'):
            raise TypeError('Expected RTDETRDecoder with input_proj and enc_score_head')
        hooks = []
        self._shapes = []
        self._dense_logits = None
        def catch_score(_module, _inputs, out):
            self._dense_logits = out
        def catch_shape(_module, _inputs, out):
            self._shapes.append((int(out.shape[-2]), int(out.shape[-1])))
        try:
            hooks.append(head.enc_score_head.register_forward_hook(catch_score))
            for proj in head.input_proj:
                hooks.append(proj.register_forward_hook(catch_shape))
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        try:
            original = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            boxes, scores = preds
            # Matcher for the final decoder layer; standard assignment is unchanged.
            matches = self.matcher(boxes[-1].detach(),scores[-1].detach(),
                                   batch['bboxes'],batch['cls'],batch['gt_groups'])
            if self._shapes:
                if len(self._shapes) != 3 or any(
                    self._shapes[i] != tuple(max(1, v // (2 ** i)) for v in self._shapes[0])
                    for i in (1, 2)
                ):
                    raise ValueError(f'Expected P3/P4/P5 feature sizes with strides 8/16/32, got {self._shapes}')
                # Neck outputs P3,P4,P5, default P3 stride 8 in R18.
                image_hw = (self._shapes[0][0]*8,self._shapes[0][1]*8)
            else:
                image_hw = (640,640) # non-GAP loss only; fixed R18 640 training profile
            addition = self.signals(boxes,scores,batch,matches,self._dense_logits,
                                    self._shapes,image_hw=image_hw)
            self.last_extra = {k:float(v.detach().item()) for k,v in addition.items()}
            return {**original, **addition}
        finally:
            # Avoid retaining a graph after optimizer step / training error.
            self._dense_logits = None
            self._shapes = []
