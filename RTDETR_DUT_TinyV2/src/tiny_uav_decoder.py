"""Tiny-UAV-aware RT-DETR decoder; opt-in, original state_dict parameter names.

Designed against lishun-lishun/RTDETR-main main at commit
9636869af908610c7aa84c4cfca3f2e33479af41 (2026-11-16 inspection).
"""
import torch
import torch.nn.functional as F
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.models.utils.ops import get_cdn_group
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class TinyUAVDecoder(RTDETRDecoder):
    """Replace RTDETRDecoder with compatible signature / outputs.

    M1: local peak-aware encoder top-k ranking. No extra learnable parameters.
    M2: pixel-conditioned contrastive denoising. Training-only.
    """

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300,
                 ndp=4, nh=8, ndl=6, *args, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, *args, **kwargs)
        self.peak_alpha = 0.0
        self.peak_kernel = 3
        self.dn_pos_px = 0.0
        self.dn_neg_px = 0.0
        self.input_pixels = 640

    def configure(self, settings):
        """Read opt-in YAML metadata after parent DetectionModel is built."""
        self.peak_alpha = float(settings.get('peak_alpha', 0.0))
        self.peak_kernel = int(settings.get('peak_kernel', 3))
        self.dn_pos_px = float(settings.get('dn_pos_px', 0.0))
        self.dn_neg_px = float(settings.get('dn_neg_px', 0.0))
        self.input_pixels = int(settings.get('input_pixels', 640))
        if self.peak_kernel < 3 or self.peak_kernel % 2 != 1:
            raise ValueError('peak_kernel must be odd and >= 3')
        if min(self.dn_pos_px, self.dn_neg_px, self.peak_alpha) < 0:
            raise ValueError('tiny_uav settings must be nonnegative')

    def _peak_rank(self, scores, shapes):
        """Enc-score logit plus a bounded positive local-contrast bonus.

        Avoids a hard per-level SAQS quota. Does not alter class scores
        returned to the loss; only the encoder query seed ranking.
        """
        if self.peak_alpha == 0:
            return scores
        result = []
        off = 0
        for h, w in shapes:
            n = int(h * w)
            src = scores[:, off: off + n].reshape(scores.shape[0], 1, h, w)
            prob = src.sigmoid()
            avg = F.avg_pool2d(prob, self.peak_kernel, stride=1,
                               padding=self.peak_kernel // 2, count_include_pad=False)
            # Clamp protects the score distribution from a very strong bonus.
            gain = (prob - avg).clamp(min=0, max=0.5)
            result.append((src + self.peak_alpha * gain).reshape(scores.shape[0], n))
            off += n
        if off != scores.shape[1]:
            raise RuntimeError('query rank score and feature shapes mismatch')
        return torch.cat(result, 1)

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        if self.peak_alpha == 0:
            return super()._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        bs, _, hd = feats.shape
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype,
                                                      device=feats.device)
        features = self.enc_output(valid_mask * feats)
        scores = self.enc_score_head(features)
        rank = self._peak_rank(scores.max(-1).values, shapes)
        topk = torch.topk(rank, self.num_queries, dim=1).indices
        qidx = topk.unsqueeze(-1)
        top_features = features.gather(1, qidx.expand(-1, -1, hd))
        anchor_sel = anchors.expand(bs, -1, -1).gather(1, qidx.expand(-1, -1, 4))
        refs = self.enc_bbox_head(top_features) + anchor_sel
        enc_boxes = refs.sigmoid()
        enc_scores = scores.gather(1, qidx.expand(-1, -1, self.nc))
        embeddings = (self.tgt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
                      if self.learnt_init_query else top_features)
        if self.training:
            refs = refs.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_bbox is not None:
            refs = torch.cat([dn_bbox, refs], 1)
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refs, enc_boxes, enc_scores

    def _pixel_dn(self, dn_boxes, batch, dn_meta):
        """Replace valid DN references with pixel-bounded box jitter.

        Uses original DN layout from get_cdn_group; positive/negative slots
        retain exactly the same indexes and attention mask. No GT at inference.
        """
        if dn_boxes is None or dn_meta is None:
            return dn_boxes
        out = dn_boxes.clone()
        bs = len(batch['gt_groups'])
        gsize = max(batch['gt_groups'])
        groups = int(dn_meta['dn_num_group'])
        negative_base = groups * gsize
        b_idx = batch['batch_idx'].view(-1)
        gts = batch['bboxes']
        for b in range(bs):
            originals = gts[b_idx == b]
            if originals.numel() == 0:
                continue
            pos_idx = dn_meta['dn_pos_idx'][b].to(device=out.device, dtype=torch.long)
            ref = originals.repeat(groups, 1).to(out.device)
            if len(pos_idx) != len(ref):
                raise RuntimeError('DN indexing no longer matches upstream get_cdn_group')
            for ids, amount in ((pos_idx, self.dn_pos_px),
                                (pos_idx + negative_base, self.dn_neg_px)):
                if amount <= 0:
                    continue
                noise = (torch.rand_like(ref) * 2.0 - 1.0) * (amount / self.input_pixels)
                # Boxes are normalized cx/cy/w/h; width and height jitter
                # is capped to avoid negative/empty boxes.
                noise[:, 2:] = noise[:, 2:].clamp(min=-0.4 * ref[:, 2:],
                                                 max=0.4 * ref[:, 2:])
                noisy = ref + noise
                corners = xywh2xyxy(noisy).clamp(0.0, 1.0)
                rebuilt = xyxy2xywh(corners).clamp(1e-6, 1.0 - 1e-6)
                out[b, ids] = torch.logit(rebuilt, eps=1e-6)
        return out

    def forward(self, x, batch=None):
        # Using original path when M2 disabled is essential for checkpoint
        # compatibility and identical random-number / DN behavior.
        if (not self.training) or (self.dn_pos_px <= 0 and self.dn_neg_px <= 0):
            return super().forward(x, batch)

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
            self.num_denoising, self.label_noise_ratio, self.box_noise_scale,
            self.training)
        dn_bbox = self._pixel_dn(dn_bbox, batch, dn_meta)
        embed, ref, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores = self.decoder(
            embed, ref, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask)
        return dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
