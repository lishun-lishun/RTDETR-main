"""GT-derived SCRE/SPLF auxiliary losses in augmented 640-input coordinates."""

import torch
import torch.nn.functional as F


def make_aux_targets(batch, h, w, small_max_px=16.0):
    """Return CenterNet-style targets, valid negatives, protection mask, GT centers.

    Ultralytics ``batch['bboxes']`` are normalized xywh *after* letterbox and
    augmentation. Pixel sizes are therefore derived from ``batch['img']``.
    Medium/large box regions are ignored in the center negative loss and kept
    protected in the feature residual penalty.
    """
    image = batch["img"]
    b = image.shape[0]
    device = image.device
    heat = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
    valid = torch.ones_like(heat)
    protect = torch.ones_like(heat)
    centers = [[] for _ in range(b)]
    large_rects = [[] for _ in range(b)]
    box_index = batch["batch_idx"].to(device=device, dtype=torch.long).flatten()
    boxes = batch["bboxes"].to(device=device, dtype=torch.float32).reshape(-1, 4)
    ih, iw = image.shape[-2:]
    for box, image_index in zip(boxes, box_index):
        bi = int(image_index.item())
        if not 0 <= bi < b:
            continue
        cx, cy, bw, bh = box.tolist()
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        x0 = max(0, min(w, int((cx - bw / 2) * w)))
        x1 = max(0, min(w, int((cx + bw / 2) * w + 1)))
        y0 = max(0, min(h, int((cy - bh / 2) * h)))
        y1 = max(0, min(h, int((cy + bh / 2) * h + 1)))
        if max(bw * iw, bh * ih) <= small_max_px:
            gx, gy = cx * w, cy * h
            ix, iy = min(w - 1, int(gx)), min(h - 1, int(gy))
            centers[bi].append((gx, gy))
            # A compact Gaussian centered on the quantized feature cell.
            radius = max(1, min(3, round(max(bw * w, bh * h) / 2)))
            yy = torch.arange(max(0, iy - 3 * radius), min(h, iy + 3 * radius + 1), device=device)
            xx = torch.arange(max(0, ix - 3 * radius), min(w, ix + 3 * radius + 1), device=device)
            gaussian = torch.exp(-((yy[:, None] - iy).square() + (xx[None, :] - ix).square()) /
                                 (2.0 * radius * radius))
            region = heat[bi, 0, yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1]
            heat[bi, 0, yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1] = torch.maximum(region, gaussian)
            # Allow the small box plus one P3-cell margin to change.
            protect[bi, 0, max(0, y0 - 1):min(h, y1 + 1),
                    max(0, x0 - 1):min(w, x1 + 1)] = 0
        else:
            large_rects[bi].append((x0, y0, x1, y1))
            valid[bi, 0, y0:y1, x0:x1] = 0
    for bi, rects in enumerate(large_rects):
        for x0, y0, x1, y1 in rects:
            protect[bi, 0, y0:y1, x0:x1] = 1
    valid = torch.maximum(valid, (heat == 1).float())
    gt_centers = [torch.tensor(items, device=device, dtype=torch.float32).reshape(-1, 2) for items in centers]
    return heat, valid, protect, gt_centers


def center_focal_loss(logits, heat, valid):
    """Numerically stable Gaussian focal loss, with no-positive images handled."""
    logits = logits.float()
    p = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    positive = heat.eq(1)
    negative_weight = (1 - heat).pow(4) * valid * (~positive)
    positive_loss = -(1 - p).pow(2) * F.logsigmoid(logits) * positive
    negative_loss = -p.pow(2) * F.logsigmoid(-logits) * negative_weight
    num_pos = positive.sum()
    denom = torch.where(num_pos > 0, num_pos, valid.sum().clamp_min(1))
    return (positive_loss.sum() + negative_loss.sum()) / denom


def matched_offset_loss(indices, offsets, gt_centers, w, max_distance=1.0):
    """Greedy one-to-one GT/candidate match; unmatched Top-K points are ignored."""
    losses = []
    for bi, gt in enumerate(gt_centers):
        if not len(gt):
            continue
        candidates = torch.stack(((indices[bi] % w).float() + 0.5,
                                  torch.div(indices[bi], w, rounding_mode="floor").float() + 0.5), dim=-1)
        distance = torch.cdist(gt.unsqueeze(0), candidates.unsqueeze(0)).squeeze(0)
        remaining = torch.ones(indices.shape[1], dtype=torch.bool, device=indices.device)
        for gi in distance.min(dim=1).values.argsort().tolist():
            possible = distance[gi].masked_fill(~remaining, float("inf"))
            nearest = int(possible.argmin().item())
            if possible[nearest].item() > max_distance:
                continue
            remaining[nearest] = False
            target = gt[gi] - candidates[nearest]
            losses.append(F.smooth_l1_loss(offsets[bi, nearest].float(), target, reduction="mean"))
    return torch.stack(losses).mean() if losses else offsets.sum() * 0.0


def protection_loss(base, enhanced, mask):
    """Penalize *only* the new residual, not drift of the original path."""
    delta = (enhanced.float() - base.float()).abs()
    return (delta * mask).sum() / (mask.sum().clamp_min(1) * delta.shape[1])


def auxiliary_losses(batch, config, scre=None, fusion=None):
    source = scre.last if scre is not None else fusion.last
    if source is None:
        raise RuntimeError("Auxiliary forward state was not produced before loss")
    logits = source["logits"]
    if logits is None:
        raise RuntimeError("Candidate center logits are missing")
    h, w = logits.shape[-2:]
    heat, valid, protect, gt_centers = make_aux_targets(batch, h, w,
                                                         float(config.get("small_max_px", 16.0)))
    center = center_focal_loss(logits, heat, valid)
    offset = (matched_offset_loss(scre.last["indices"], scre.last["offsets"], gt_centers, w)
              if scre is not None and scre.use_offset else center * 0.0)
    if fusion is not None and fusion.last is not None:
        protected = protection_loss(fusion.last["base"], fusion.last["enhanced"], protect)
    else:
        protected = center * 0.0
    return (center * float(config.get("lambda_center", 0.0)),
            offset * float(config.get("lambda_offset", 0.0)),
            protected * float(config.get("lambda_protect", 0.0)))
