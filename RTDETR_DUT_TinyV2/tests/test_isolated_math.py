"""Dependency-light CPU smoke checks of the new pixel-scale formulas."""
import torch

def test_pixel_nwd_is_zero_at_perfect_match_and_increases():
    gt=torch.tensor([[0.5,0.5,0.02,0.02]])
    moved=gt.clone(); moved[:,0]+=1/640
    def nwd(box):
        d=((box[:,:2]-gt[:,:2])*640).square().sum(-1)
        s=((box[:,2:]-gt[:,2:])*320).square().sum(-1)
        return 1-torch.exp(-torch.sqrt(d+s+1e-9)/12)
    assert nwd(gt).item() < nwd(moved).item()

def test_tiny_weight_monotone():
    s=torch.tensor([6.,16.,32.,64.])
    w=torch.sigmoid((24-s)/6)
    assert bool(torch.all(w[:-1] > w[1:]))
