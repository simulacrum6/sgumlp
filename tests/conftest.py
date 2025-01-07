import pytest
import torch


@pytest.fixture
def hs_image():
    return torch.randn(2, 16, 128, 128)
