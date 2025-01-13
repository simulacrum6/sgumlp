import torch

from src.models import MLPMixer, SGUMLPMixer


def test_MLPMixer(hs_image):
    mixer = MLPMixer(
        hs_image.shape[1:],
        patch_size=16,
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
    )
    y = mixer(hs_image)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([hs_image.shape[0], mixer.n_tokens, mixer.n_channels])
    assert torch.all(torch.eq(out_shape, expected_shape))


def test_SGUMLPMixer(patches):
    mixer = SGUMLPMixer(
        patches.shape[1:],
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
    )
    y = mixer(patches)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([patches.shape[0], mixer.n_tokens, mixer.n_channels])
    assert torch.all(torch.eq(out_shape, expected_shape))

    # learnable residuals
    mixer = SGUMLPMixer(
        patches.shape[1:],
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
        residual_weight=1.0,
        learnable_residual=True,
    )
    y = mixer(patches)