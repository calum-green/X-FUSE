import pytest
import torch

DINO_FLAVOURS = [
    {
        "id": "dinov2_s_regs",
        "feat_dim": 384,
        "num_heads": 6,
        "patch_size": 14,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov2_b_regs",
        "feat_dim": 768,
        "num_heads": 12,
        "patch_size": 14,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov2_l_regs",
        "feat_dim": 1024,
        "num_heads": 16,
        "patch_size": 14,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov2_g_regs",
        "feat_dim": 1536,
        "num_heads": 16,
        "patch_size": 14,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov2_s_no_regs",
        "feat_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "n_register_tokens": 0,
    },
    {
        "id": "dinov3_s_regs",
        "feat_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov3_s_no_regs",
        "feat_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "n_register_tokens": 0,
    },
    {
        "id": "dinov3_l_regs",
        "feat_dim": 1024,
        "num_heads": 16,
        "patch_size": 16,
        "n_register_tokens": 4,
    },
    {
        "id": "dinov3_7b_regs",
        "feat_dim": 4096,
        "num_heads": 32,
        "patch_size": 16,
        "n_register_tokens": 4,
    },
]


@pytest.fixture(params=DINO_FLAVOURS, ids=[f["id"] for f in DINO_FLAVOURS])
def dino_flavour(request):
    return request.param


@pytest.fixture
def features(dino_flavour):
    torch.manual_seed(0)
    return torch.randn(1, dino_flavour["feat_dim"], 8, 8)


@pytest.fixture
def xrd_map_4d():
    torch.manual_seed(42)
    return torch.rand(1, 1, 8, 8)
