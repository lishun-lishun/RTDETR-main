"""Run in original RTDETR-main root after installer with pytest -q ..."""
import torch
import pytest
from ultralytics.nn.modules.tiny_uav_decoder import TinyUAVDecoder
from ultralytics.models.utils.tiny_uav_loss import TinyUAVRTDETRLoss
from ultralytics.nn.tasks import RTDETRDetectionModel


def test_decoder_parameter_keys_identical():
    from ultralytics.nn.modules.head import RTDETRDecoder
    a = RTDETRDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3)
    b = TinyUAVDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3)
    assert a.state_dict().keys() == b.state_dict().keys()


def test_tiny_model_fwd_eval():
    from pathlib import Path
    yaml = Path('ultralytics/cfg/models/rt-detr/rtdetr_r18_tinyv2_m1_peak.yaml')
    assert yaml.exists()
    model = RTDETRDetectionModel(str(yaml), nc=1, verbose=False).eval()
    with torch.no_grad():
        y = model(torch.rand(1, 3, 128, 128))
    assert y[0].shape == (1, 300, 5)
    assert torch.isfinite(y[0]).all()


def test_loss_matching_and_grad():
    config = {'nwd_weight': 1., 'center_match_weight': .6, 'input_pixels': 640}
    fn = TinyUAVRTDETRLoss(nc=1, settings=config, use_vfl=True)
    boxes = torch.tensor([[[.51,.5,.02,.03], [.1,.1,.05,.05], [.8,.8,.1,.1]]],requires_grad=True)
    scores = torch.tensor([[[1.], [-1.], [-1.]]], requires_grad=True)
    pred_boxes = boxes.unsqueeze(0).repeat(2,1,1,1)
    pred_scores = scores.unsqueeze(0).repeat(2,1,1,1)
    batch = {'cls': torch.tensor([0]), 'bboxes':torch.tensor([[.5,.5,.02,.03]]),
             'gt_groups':[1]}
    losses = fn((pred_boxes,pred_scores),batch)
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_train_forward_dn():
    head = TinyUAVDecoder(1, (256,256,256),256,300,4,8,3)
    head.configure({'dn_pos_px':1.25,'dn_neg_px':4.,'input_pixels':640})
    head.train()
    inp = [torch.rand(1,256,16,16), torch.rand(1,256,8,8), torch.rand(1,256,4,4)]
    batch = {'cls':torch.tensor([0]),'bboxes':torch.tensor([[.5,.5,.02,.02]]),
             'batch_idx':torch.tensor([0]),'gt_groups':[1]}
    a,b,c,d,dn = head(inp,batch)
    assert dn is not None and a.shape[2] == 300 + dn['dn_num_split'][0]
    assert a.isfinite().all() and b.isfinite().all()


def test_off_control_matches_original_and_checkpoint(tmp_path):
    from ultralytics.nn.modules.head import RTDETRDecoder

    torch.manual_seed(7)
    original = RTDETRDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3).eval()
    changed = TinyUAVDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3).eval()
    changed.load_state_dict(original.state_dict(), strict=True)
    features = [torch.randn(1, 256, s, s) for s in (16, 8, 4)]
    with torch.no_grad():
        expected = original(features)[0]
        actual = changed(features)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checkpoint = tmp_path / 'decoder.pt'
    torch.save(changed.state_dict(), checkpoint)
    reloaded = TinyUAVDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3).eval()
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(features)[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize('groups', ([0, 0], [1, 0], [1, 2]))
def test_dn_batch_two_and_empty_gt(groups):
    head = TinyUAVDecoder(1, (256, 256, 256), 256, 300, 4, 8, 3)
    head.configure({'dn_pos_px': 1.25, 'dn_neg_px': 4.0, 'input_pixels': 640})
    head.train()
    n = sum(groups)
    batch = {
        'cls': torch.zeros(n, dtype=torch.long),
        'bboxes': torch.tensor([[.5, .5, .02, .02]] * n).reshape(n, 4),
        'batch_idx': torch.tensor([i for i, count in enumerate(groups) for _ in range(count)]),
        'gt_groups': list(groups),
    }
    features = [torch.rand(2, 256, s, s) for s in (16, 8, 4)]
    boxes, scores, enc_boxes, enc_scores, meta = head(features, batch)
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert enc_boxes.shape == (2, 300, 4) and enc_scores.shape == (2, 300, 1)
    if n:
        assert boxes.shape[2] == 300 + meta['dn_num_split'][0]
        for count, pos in zip(groups, meta['dn_pos_idx']):
            assert len(pos) == count * meta['dn_num_group']
            assert (pos < meta['dn_num_split'][0]).all()
        from ultralytics.models.utils.ops import get_cdn_group
        _, _, mask, direct_meta = get_cdn_group(
            batch, 1, 300, head.denoising_class_embed.weight,
            head.num_denoising, head.label_noise_ratio,
            head.box_noise_scale, training=True)
        dn_count = direct_meta['dn_num_split'][0]
        assert mask.shape == (dn_count + 300, dn_count + 300)
        assert mask[dn_count:, :dn_count].all()
        assert not mask[dn_count:, dn_count:].any()
    else:
        assert meta is None and boxes.shape[2] == 300


def test_nwd_one_pixel_shift_and_gradient():
    fn = TinyUAVRTDETRLoss(nc=1, settings={'nwd_weight': 1.0})
    fn.device = torch.device('cpu')
    gt = torch.tensor([[.5, .5, .02, .02]])
    exact = fn._get_loss_bbox(gt, gt)['loss_giou']
    shifted = gt.clone().requires_grad_(True)
    shifted.data[:, 0] += 1 / 640
    moved = fn._get_loss_bbox(shifted, gt)['loss_giou']
    assert moved > exact
    moved.backward()
    assert shifted.grad is not None and torch.isfinite(shifted.grad).all()
