"""Run in the repository root with the server's PyTorch environment."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel


YAML = Path('ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_gap.yaml')


@pytest.mark.parametrize('count', [0, 1, 3])
def test_train_step_hook_cleanup_and_serialization(count, tmp_path):
    model = RTDETRDetectionModel(str(YAML), nc=1, verbose=False)
    model.train()
    gt = torch.tensor([[.5, .5, .02, .02]] * count).reshape(count, 4)
    batch = {
        'img': torch.rand(1, 3, 128, 128),
        'cls': torch.zeros(count, 1, dtype=torch.long),
        'bboxes': gt,
        'batch_idx': torch.zeros(count, dtype=torch.long),
    }
    head = model.model[-1]
    before = (len(head.enc_score_head._forward_hooks),
              [len(p._forward_hooks) for p in head.input_proj])
    loss, parts = model(batch)
    assert torch.isfinite(loss) and torch.isfinite(parts).all()
    assert model.criterion.last_extra['loss_gap'] >= 0
    assert len(head.enc_score_head._forward_hooks) == before[0]
    assert [len(p._forward_hooks) for p in head.input_proj] == before[1]
    loss.backward()
    assert head.enc_score_head.weight.grad is not None
    assert torch.isfinite(head.enc_score_head.weight.grad).all()
    clone = deepcopy(model)
    save_path = tmp_path / 'model.pt'
    torch.save({'model': clone}, save_path)
    restored = torch.load(save_path, map_location='cpu', weights_only=False)['model']
    assert len(restored.model[-1].enc_score_head._forward_hooks) == 0


def test_eval_has_no_hook():
    model = RTDETRDetectionModel(str(YAML), nc=1, verbose=False).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 128, 128))
    assert out[0].shape[1] == 300
    assert len(model.model[-1].enc_score_head._forward_hooks) == 0
