import pytest
import torch


@pytest.fixture
def hs_image():
    return torch.randn(2, 16, 128, 128)


@pytest.fixture
def tokens():
    return torch.randn(2, 81, 768)


@pytest.fixture
def image_and_labels():
    image = torch.rand((128, 128, 16)).numpy()
    labels = torch.randint(0, 10, (128, 128)).long().numpy()
    return image, labels


@pytest.fixture
def patches():
    return torch.randn(768, 9, 9, 32)
