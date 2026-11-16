"""Full model training-step checks for the configured CUDA server."""
from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import RTDETRDetectionModel


@pytest.fixture(scope='module')
def model():
    if not torch.cuda.is_available():
        pytest.skip('CUDA runtime unavailable')
    path = Path('ultralytics/cfg/models/rt-detr/rtdetr_r18_tinyv2_all4.yaml')
    return RTDETRDetectionModel(str(path), nc=1, verbose=False).cuda().train()


@pytest.mark.parametrize('batch_size,count,amp', [
    (1, 0, False), (1, 1, False), (2, 3, False),
    (1, 0, True), (1, 1, True), (2, 3, True),
])
def test_full_model_training_step(model, batch_size, count, amp):
    batch = {
        'img': torch.rand(batch_size, 3, 128, 128, device='cuda'),
        'cls': torch.zeros(count, 1, dtype=torch.long, device='cuda'),
        'bboxes': torch.tensor([[.5, .5, .02, .02]] * count,
                              dtype=torch.float32, device='cuda').reshape(count, 4),
        'batch_idx': torch.tensor([i * batch_size // count for i in range(count)],
                                  dtype=torch.long, device='cuda'),
    }
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        loss, parts = model(batch)
    assert torch.isfinite(loss) and torch.isfinite(parts).all()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
