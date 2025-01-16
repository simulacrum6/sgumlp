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


@pytest.fixture
def dataset():
    n = 64
    p = 17
    c = 32
    k = 10
    X = torch.randn(n, p, p, c)
    y = torch.randint(0, k, (n,)).long()
    return X, y, k


@pytest.fixture
def sgumlpmixer_args():
    return dict(
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
    )


@pytest.fixture
def adamw_args():
    return dict(
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
