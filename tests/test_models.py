import torch

from src.models import MLPMixer, SGUMLPMixer

def test_MLPMixer(hs_image):
    mixer = MLPMixer(
        hs_image.shape[1:],
        patch_size=16,
        token_features=256,
        mixer_mlp_hidden_features=768,
        num_blocks=2,
        activation="relu",
    )
    y = mixer(hs_image)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([hs_image.shape[0], mixer.n_tokens, mixer.n_channels])
    assert torch.all(torch.eq(out_shape, expected_shape))


def test_SGUMLPMixer(hs_image):
    mixer = SGUMLPMixer(
        hs_image.shape[1:],
        patch_size=16,
        token_features=256,
        mixer_mlp_hidden_features=768,
        num_blocks=2,
        activation="relu",
    )
    y = mixer(hs_image)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([hs_image.shape[0], mixer.n_tokens, mixer.n_channels])
    assert torch.all(torch.eq(out_shape, expected_shape))
